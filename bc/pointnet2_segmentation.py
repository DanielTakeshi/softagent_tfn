# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py
import os.path as osp
import numpy as np
from bc.pointnet2_classification import GlobalSAModule, SAModule
from bc.se3 import flow2pose

import torch
import torch.nn.functional as F
from pytorch3d.transforms import (
    Rotate, matrix_to_quaternion, quaternion_to_axis_angle
)
from torch_scatter import scatter
from torchmetrics.functional import jaccard_index
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, knn_interpolate
torch.set_printoptions(sci_mode=False, precision=6)


class FPModule(torch.nn.Module):
    """This module is used in segmentation (but not classification) with PointNet++.

    Notes on the forward pass of FP module...

    Daniel: input consists of two pairs of (x, pos, batch), one for the current
    most recent tensor being propagated through the network, and the second is
    a skip connection from a prior layer. We do use both of these, so we will
    concatenate and then run a neural network, specified with MLP when defining
    the PointNet2. The dimensions per point by default go from:
        3+feat_dim -> 3+128 -> 3+256 -> 3+1024
    This will be used to go in 'reverse' due to subsampling points. First we go
    from 1024-dim input with 256-dim skip, so the MLP must start with 1024+256
    nodes, and produces 256-dim features (`fp3_module`). That gets combined with
    128-dim from the skip connection, so MLP starts with 256+128 and produces 128
    dim output (`fp2_module`). Finally, `fp1_module` takes that 128-dim output,
    and the ORIGINAL INPUT as skip connection, but ONLY `data.x`, NOT `data.pos`.
    That means we need 128 + (dim of data.x) to start the MLP.
    """

    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet2_Flow(torch.nn.Module):
    """PointNet++ architectue for flow prediction.

    Should be the same as the segmentation version, except as with the analogue
    to the classification case, we don't use log softmax at the end. It directly
    predicts flow, returning (N,flow_dim) where typically `flow_dim=3` and N is
    all the points in the minibatch (might not be same # of points per PCL).
    There may be further processing such as differentiable, parameter-less SVD.

    04/18/2022: changing radius hyperparameter to make it consistent w/Regression.
    04/20/2022: can output flow, or compress the flow into a pose.
    05/10/2022: minibatches can have non-equal numbers of points per PCL.
        Requires each data to have the `ptr` to indicate idxs for each PCL.
    05/16/2022: support 6D flow.
    05/21/2022: support dense transformation policy.
    """

    def __init__(self, in_dim, encoder_type=None, scale_pcl_val=None):
        super().__init__()
        self.in_dim = in_dim  # 3 for pos, then rest for segmentation
        self.flow_dim = 6 if '6d_flow' in encoder_type else 3
        self.encoder_type = encoder_type
        self.scale_pcl_val = scale_pcl_val

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([in_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # The `in_dim-3` is because we only want the `data.x` part, not `data.pos`.
        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + (in_dim-3), 128, 128, 128]))

        self.mlp = MLP([128, 128, 128, self.flow_dim], dropout=0.5, batch_norm=False)

    def forward(self, data, info=None):
        """Forward pass, store the flow, potentially do further processing."""

        # Special case if we scaled everything (e.g., meter -> millimeter), need
        # to downscale the observation because otherwise PN++ produces NaNs. :(
        if self.scale_pcl_val is not None:
            data.pos /= self.scale_pcl_val

        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        # Unlike in segmentation, don't apply a `log_softmax(dim=-1)`.
        flow_per_pt = self.mlp(x)  # this is per-point, of size (N,flow_dim)
        self.flow_per_pt = flow_per_pt  # use for flow visualizations
        self.flow_per_pt_r = None # only used for 6D flow

        # Only used for 6D flow consistency loss, detect otherwise with `None`
        self.means = None
        self.rot_flows = None

        # Must revert `data.pos` back to the original scale before SVD!!
        if self.scale_pcl_val is not None:
            data.pos *= self.scale_pcl_val

        # Handle different encoder types.
        if self.encoder_type == 'pointnet':
            # Per-point flow, with NON-differentiable averaging or SVD later.
            return flow_per_pt

        elif self.encoder_type == 'pointnet_avg':
            # Careful, apply averaging to separate minibatch items!
            # Not vectorizing, but this should be correct.
            acts = []
            for i in range(len(data.ptr)-1):
                idx1 = data.ptr[i].detach().cpu().numpy().item()
                idx2 = data.ptr[i+1].detach().cpu().numpy().item()
                flow_one = flow_per_pt[idx1:idx2]  # just this PCL's flow.
                #! Change the index to reflect the one hot encoding of the tool
                tool_one = torch.where(data.x[idx1:idx2, 0] == 1)[0]  # tool 0-th col
                mean_one = torch.mean(flow_one[tool_one], dim=0)  # only tool!
                acts.append(mean_one)
            # Using a fixed (1,3), assuming this is translation only.
            acts_all = torch.cat([a.reshape(1,3) for a in acts])
            return acts_all.squeeze()

        elif self.encoder_type == 'pointnet_svd_pointwise':
            # Given predicted flow, run SVD, apply transform, get difference.
            flows = []
            for i in range(len(data.ptr)-1):
                idx1 = data.ptr[i].detach().cpu().numpy().item()
                idx2 = data.ptr[i+1].detach().cpu().numpy().item()
                flow_one = flow_per_pt[idx1:idx2]  # just this PCL's flow.
                posi_one = data.pos[idx1:idx2]  # just this PCL's xyz (tool+items).
                tool_one = torch.where(data.x[idx1:idx2, 0] == 1)[0]  # tool 0-th col
                xyz  = posi_one[tool_one]
                flow = flow_one[tool_one]

                # Get a _transform_ to apply on the original tool pos.
                trfm = flow2pose(
                        xyz=xyz[None,:],
                        flow=flow[None,:],
                        weights=None,
                        return_transform3d=True,
                        return_quaternions=False,
                )

                # Applies on ALL xyz points, but later in loss fxn, we filter by tool.
                trfm_xyz = trfm.transform_points(posi_one).squeeze(0)
                batch_flow = trfm_xyz - posi_one
                flows.append(batch_flow)

            return torch.cat(flows)

        elif self.encoder_type == 'pointnet_svd_pointwise_6d_flow':
            # From Dave: could be worth trying producing a 6D output, where 3
            # dimensions are averaged to get the translation and the other 3 are
            # input to SVD to get rotation.

            # Handle two architecture variants; split transl. and rotat. parts.
            # Also assign to these for the consistency loss later (if desired).
            self.flow_per_pt = flow_per_pt[:, :3]
            self.flow_per_pt_r = flow_per_pt[:, 3:]

            # Right now, doing this with pointwise loss.
            flows = []

            # Tracking for consistency loss
            self.means = []
            self.rot_flows = []
            for i in range(len(data.ptr)-1):
                #* This is iterating through the minibatch
                #* Getting the start and the ending indices of the minibatch
                idx1 = data.ptr[i].detach().cpu().numpy().item()
                idx2 = data.ptr[i+1].detach().cpu().numpy().item()

                #* Getting the position of all the points from the current data batch, that belong to the
                #* current mini-batch
                posi_one = data.pos[idx1:idx2]  # just this PCL's xyz (tool+items).

                #* Finding specifically the tool points here. Check the git blame, I've changed the indices to the
                #* approproate values
                tool_one = torch.where(data.x[idx1:idx2, 0] == 1)[0]  # tool 0-th col

                #* We get flow_per_pt and flow_per_pt_r from a single vector or two seperate vectors if
                #* we set seperate_r_t MLPs. We now grab the translation and rotation flow for all the points in the current mini-batch
                # Just this PCL's flow, but split into translation and rotation.
                flow_one_t = self.flow_per_pt[idx1:idx2]
                flow_one_r = self.flow_per_pt_r[idx1:idx2]

                #* For translations we take the mean of the translation predicted by the self.mlp layer above, but only for the tool points
                # Translation is just the average of the _tool_ flow.
                mean_one = torch.mean(flow_one_t[tool_one], dim=0) # (3,)

                #* We take only the xyz of the tool and the flow corresponding to these tool points
                # We get rotation by passing tool flow to SVD.
                xyz  = posi_one[tool_one]
                flow = flow_one_r[tool_one]  # note the `r`

                #* We extract the rotation and translation matrices from flow2pose, we do not use translation since we already extracted it above
                # Get a _transform_ to apply on the original tool pos.
                rot_matrices, _ = flow2pose(
                        xyz=xyz[None,:],
                        flow=flow[None,:],
                        weights=None,
                        return_transform3d=False,
                        return_quaternions=False,
                        world_frameify=False,
                )

                #* Finding the mean of the xyz position
                # Correct rotation frame
                xyz_mean = xyz.unsqueeze(0).mean(dim=1, keepdims=True)
                #* subjecting that point to a transformation from the flow2pose function
                t_correction = (xyz_mean - torch.bmm(xyz_mean, rot_matrices)).squeeze(1) # (1, 3)

                #* Final transformation, using previously computed translation from the averaging of flow_pt_t, t_correction and the rotation
                # Compute a transformation.
                trfm = Rotate(rot_matrices).translate(mean_one[None,:] + t_correction)

                #* Applying the transformation to all the points
                # Applies on ALL xyz points, but later in loss fxn, we filter by tool.
                trfm_xyz = trfm.transform_points(posi_one).squeeze(0)

                #* Computing the flow to all points, but loss only for tool points. So now, the network received (N, 3) pointcloud as input
                #* generated a 6D vector of shape (N, 6) (or two (N, 3) vectors, if we use seperate MLPs), from which we pick only the tool points. Then we mean the [:3] of the vector
                batch_flow = trfm_xyz - posi_one
                # print('batch_flow.shape: {}'.format(batch_flow.size()))
                flows.append(batch_flow)

                # Consistency loss flow computations
                # TODO: Maybe don't do this if we don't need to?
                post_flow_t = torch.zeros_like(flow_one_t)
                post_flow_t += mean_one
                self.means.append(post_flow_t)

                trfm_only_r = Rotate(rot_matrices).translate(t_correction)
                post_flow_r = trfm_only_r.transform_points(posi_one).squeeze(0) - posi_one
                self.rot_flows.append(post_flow_r)

            # Combine consistency loss targets
            #* Collection of all flow translations
            self.means = torch.cat(self.means)
            #* Collection of all flow rotations
            self.rot_flows = torch.cat(self.rot_flows)
            #* Returning the flow
            return torch.cat(flows)

        else:
            raise ValueError(self.encoder_type)
