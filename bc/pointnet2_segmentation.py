# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py
from bc.pointnet2_classification import GlobalSAModule, SAModule
from bc.se3 import flow2pose
import torch
from pytorch3d.transforms import (
    Rotate, matrix_to_quaternion, quaternion_to_axis_angle,
    axis_angle_to_matrix,
)
from torch_geometric.nn import MLP, knn_interpolate
torch.set_printoptions(sci_mode=False, precision=6)


class FPModule(torch.nn.Module):
    """This module is used in segmentation (but not classification) with PointNet++."""

    def __init__(self, k, nn, use_skip=True):
        super().__init__()
        self.k = k
        self.nn = nn
        self.use_skip = use_skip

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if (x_skip is not None) and self.use_skip:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet2_Segm(torch.nn.Module):
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
    06/02/2022: slight improvement to baseline with clear masks.
    06/05/2022: support removing skip connections. Technically this will use skip
        for the interpolation stage, but it just won't do the torch.cat([x,x_skip])
        which seems like that's more in the spirit of testing this ablation.
    """

    def __init__(self, in_dim, flow_dim=3, encoder_type=None, scale_pcl_val=None,
            separate_MLPs_R_t=False, dense_transform=False,
            remove_skip_connections=False):
        super().__init__()
        self.in_dim = in_dim  # 3 for pos, then rest for segmentation
        self.flow_dim = flow_dim
        self.encoder_type = encoder_type
        self.scale_pcl_val = scale_pcl_val
        self.separate_MLPs_R_t = separate_MLPs_R_t
        if 'dense_tf_3D_' in self.encoder_type:
            assert self.flow_dim == 3, self.flow_dim
        if self.separate_MLPs_R_t or ('dense_tf_6D_' in self.encoder_type):
            assert self.flow_dim == 6, self.flow_dim
        self.dense_transform = dense_transform
        self.remove_skip_connections = remove_skip_connections
        self._mask = None

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([in_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # The `in_dim-3` is because we only want the `data.x` part, not `data.pos`.
        # If removing skip connections, change # of nodes in the first MLP layer.
        if self.remove_skip_connections:
            self.fp3_module = FPModule(1, MLP([1024, 256, 256]), use_skip=False)
            self.fp2_module = FPModule(3, MLP([ 256, 256, 128]), use_skip=False)
            self.fp1_module = FPModule(3, MLP([ 128, 128, 128, 128]), use_skip=False)
        else:
            self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
            self.fp2_module = FPModule(3, MLP([ 256 + 128, 256, 128]))
            self.fp1_module = FPModule(3, MLP([ 128 + (in_dim-3), 128, 128, 128]))

        if self.separate_MLPs_R_t:
            self.mlp_t = MLP([128, 128, 128, 3], dropout=0.5, batch_norm=False)
            self.mlp_R = MLP([128, 128, 128, 3], dropout=0.5, batch_norm=False)
        else:
            self.mlp = MLP([128, 128, 128, self.flow_dim], dropout=0.5, batch_norm=False)

    def assign_clear_mask(self, mask):
        """
        For pouring or scooping, in case we have a 6D vector and need to extract
        a transformation from it here (and want to clear out unused components).
        """
        self._mask = mask

    def forward(self, data, info=None, epoch=None, rgt=None):
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
        if self.separate_MLPs_R_t:
            flow_t = self.mlp_t(x)
            flow_R = self.mlp_R(x)
        else:
            flow_per_pt = self.mlp(x)  # this is per-point, of size (N,flow_dim)
            self.flow_per_pt = flow_per_pt  # use for flow visualizations
            self.flow_per_pt_r = None # only used for 6D flow

        # Only used for 6D flow consistency loss, detect otherwise with `None`
        self.means = None
        self.rot_flows = None

        # Must revert `data.pos` back to the original scale before SVD!!
        if self.scale_pcl_val is not None:
            data.pos *= self.scale_pcl_val

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
                tool_one = torch.where(data.x[idx1:idx2, 0] == 1)[0]  # tool 0-th col
                mean_one = torch.mean(flow_one[tool_one], dim=0)  # only tool!
                acts.append(mean_one)
            # Using a fixed (1,3), assuming this is translation only.
            acts_all = torch.cat([a.reshape(1,3) for a in acts])
            return acts_all.squeeze()

        elif self.encoder_type == 'pointnet_svd':
            # Given predicted flow, run SVD for predicted (rotation,translation).
            # TODO(daniel): been a while since we tested, but this is basically the
            # 3D flow method with SVD but where we use MSE instead of pointwise.
            # NOTE(daniel): careful if we need transpose of rot matrices (or not).

            # Get tool tip pos for translation correction here.
            if len(info.shape) == 1:
                info = info.unsqueeze(0)
            tool_tip_pos = info[:,:3]

            # Tracking for consistency loss
            self.flows_3d = []

            # Again, not currently vectorizing ...
            acts = []
            for i in range(len(data.ptr)-1):
                tip_pos_one = tool_tip_pos[i]  # (3,)
                idx1 = data.ptr[i].detach().cpu().numpy().item()
                idx2 = data.ptr[i+1].detach().cpu().numpy().item()
                flow_one = flow_per_pt[idx1:idx2]  # just this PCL's flow.
                posi_one = data.pos[idx1:idx2]  # just this PCL's xyz (tool+items).
                tool_one = torch.where(data.x[idx1:idx2, 0] == 1)[0]  # tool 0-th col
                xyz  = posi_one[tool_one]
                flow = flow_one[tool_one]

                # The quat,trans for just this data point, just on tool points.
                rot_matrices, trans = flow2pose(
                        xyz=xyz[None,:],
                        flow=flow[None,:],
                        weights=None,
                        return_transform3d=False,
                        return_quaternions=False,
                        world_frameify=False,
                )  # Resulting shapes: (1,3,3) and (1,3), respectively.
                xyz_mean = xyz.unsqueeze(0).mean(dim=1, keepdims=True)  # (1,1,3)

                # For consistency loss purposes, following other code, following SE(3) code.
                # The `trans` here is `flow_mean` in `flow2pose` if we return a transform.
                trfm = Rotate(rot_matrices).translate(
                    (trans + xyz_mean - torch.bmm(xyz_mean, rot_matrices)).squeeze(1)
                )  # careful, squeezing following world_frameify case
                trfm_xyz = trfm.transform_points(posi_one).squeeze(0)
                batch_flow = trfm_xyz - posi_one
                self.flows_3d.append(batch_flow)

                # Convert rotation into axis-angle with transpose!
                quats = matrix_to_quaternion(matrix=rot_matrices.transpose(1,2))
                axis_ang = quaternion_to_axis_angle(quaternions=quats)

                # Correct translation w.r.t. tool tip as origin
                # NOTE (eddie): if we use weights, modify this xyz_mean computation
                relative_xyz_mean = xyz_mean - tip_pos_one[None,None,:]  # (1,1,3)
                t_correction = relative_xyz_mean - torch.bmm(relative_xyz_mean, rot_matrices)
                trans += t_correction.squeeze(1)

                pred_eepose = torch.cat([trans,axis_ang], dim=1)  # (1,6)
                acts.append(pred_eepose)

            # Concat and pull out for later.
            self.flows_3d = torch.cat(self.flows_3d)

            # For translation-only, also enforce no rotation (identity)? This
            # regresses to unit quaternion (1,0,0,0). PyTorch3D sets w to be the
            # first index, but pyflex shapes have w as the last index.
            acts_all = torch.cat([a for a in acts])
            return acts_all.squeeze()

        elif self.encoder_type == 'pointnet_svd_6d_flow_mse_loss':
            # NOTE(daniel): ablation of the 6D flow method. Use SVD to get (R,t)
            # but we just don't use pointwise, just return 6D vector directly.
            # This can also (and probably should) use consistency loss since that
            # will provide direct supervision to all points (pre-SVD) for which we
            # then apply a tool mask (in `bc/bc.py`).

            # Get origin of tool frame for translation correction here.
            if len(info.shape) == 1:
                info = info.unsqueeze(0)
            tool_tip_pos = info[:,:3]

            # Handle two architecture variants; split transl. and rotat. parts.
            # Also assign to these for the consistency loss later (if desired).
            if self.separate_MLPs_R_t:
                self.flow_per_pt = flow_t
                self.flow_per_pt_r = flow_R
            else:
                self.flow_per_pt = flow_per_pt[:, :3]
                self.flow_per_pt_r = flow_per_pt[:, 3:]

            # Will NOT use pointwise loss, but MSE on 6D vectors.
            actions = []

            # Tracking for consistency loss
            self.means = []
            self.rot_flows = []
            for i in range(len(data.ptr)-1):
                tip_pos_one = tool_tip_pos[i]  # (3,)
                idx1 = data.ptr[i].detach().cpu().numpy().item()
                idx2 = data.ptr[i+1].detach().cpu().numpy().item()
                posi_one = data.pos[idx1:idx2]  # just this PCL's xyz (tool+items).
                tool_one = torch.where(data.x[idx1:idx2, 0] == 1)[0]  # tool 0-th col

                # Just this PCL's flow, but split into translation and rotation.
                flow_one_t = self.flow_per_pt[idx1:idx2]
                flow_one_r = self.flow_per_pt_r[idx1:idx2]

                # Translation is just the average of the _tool_ flow.
                mean_one = torch.mean(flow_one_t[tool_one], dim=0) # (3,)

                # We get rotation by passing tool flow to SVD.
                xyz  = posi_one[tool_one]
                flow = flow_one_r[tool_one]  # note the `r`

                # Get a _transform_ to apply on the original tool pos.
                rot_matrices, _ = flow2pose(
                        xyz=xyz[None,:],
                        flow=flow[None,:],
                        weights=None,
                        return_transform3d=False,
                        return_quaternions=False,
                        world_frameify=False,
                )

                # Convert rotation into axis-angle with transpose!
                quats = matrix_to_quaternion(matrix=rot_matrices.transpose(1, 2))
                axis_ang = quaternion_to_axis_angle(quaternions=quats)

                # Change rotation frame to be around tool origin.
                xyz_mean = xyz.unsqueeze(0).mean(dim=1, keepdims=True) # (1,1,3)
                relative_xyz_mean = xyz_mean - tip_pos_one[None, None, :] # (1,1,3)
                t_correction = (relative_xyz_mean -
                    torch.bmm(relative_xyz_mean, rot_matrices)).squeeze(1)  # (1,3)
                translation = mean_one[None,:] + t_correction.squeeze(1) # (1,3)

                pred_eepose = torch.cat([translation, axis_ang], dim=1)  # (1,6)
                actions.append(pred_eepose)

                # Consistency loss flow (following pointnet_svd_pointwise_6d_flow)
                post_flow_t = torch.zeros_like(flow_one_t) # (N,3)
                post_flow_t += mean_one
                self.means.append(post_flow_t)

                # rot_matrices: (1,3,3), t_correction: (1,3), uses the correction so
                # that this rotates w.r.t. the tool origin, producing rotation flow.
                trfm_only_r = Rotate(rot_matrices).translate(t_correction)
                post_flow_r = trfm_only_r.transform_points(posi_one).squeeze(0) - posi_one
                self.rot_flows.append(post_flow_r)

            # Combine consistency loss targets, then return all actions.
            self.means = torch.cat(self.means)
            self.rot_flows = torch.cat(self.rot_flows)
            return torch.cat([a for a in actions]).squeeze()

        elif self.encoder_type == 'pointnet_svd_centered':
            # Given predicted flow, run SVD for predicted (rotation,translation).
            # Again, not currently vectorizing ...
            acts = []
            for i in range(len(data.ptr)-1):
                idx1 = data.ptr[i].detach().cpu().numpy().item()
                idx2 = data.ptr[i+1].detach().cpu().numpy().item()
                flow_one = flow_per_pt[idx1:idx2]  # just this PCL's flow.
                posi_one = data.pos[idx1:idx2]  # just this PCL's xyz (tool+items).
                tool_one = torch.where(data.x[idx1:idx2, 0] == 1)[0]  # tool 0-th col
                xyz  = posi_one[tool_one]
                flow = flow_one[tool_one]

                # The quat,trans for just this data point, just on tool points.
                quats, trans = flow2pose(
                        xyz=xyz[None,:],
                        flow=flow[None,:],
                        weights=None,
                        return_transform3d=False,
                        return_quaternions=True,
                        world_frameify=False,
                )  # Resulting shapes: (1,4) and (1,3), respectively.
                pred_eepose = torch.cat([trans,quats], dim=1)  # (1,7)
                acts.append(pred_eepose)

            # For translation-only, also enforce no rotation (identity)? This
            # regresses to unit quaternion (1,0,0,0). PyTorch3D sets w to be the
            # first index, but pyflex shapes have w as the last index.
            acts_all = torch.cat([a for a in acts])
            return acts_all.squeeze()

        elif self.encoder_type in ['pointnet_svd_pointwise',
                'pointnet_svd_pointwise_PW_bef_SVD']:
            # NOTE(daniel): adding the point-matching loss before SVD case here. In
            # that case we don't actually need the flows we just need the stuff from
            # the trunk; at test time we'd take the trunk and do non-different. SVD.
            # BUT, putting this case here gives us flexibility in case we also want
            # to try the consistency loss with the point-matching loss?

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

                # Applies on ALL points, but later in loss fxn, we filter by tool.
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
            if self.separate_MLPs_R_t:
                self.flow_per_pt = flow_t
                self.flow_per_pt_r = flow_R
            else:
                self.flow_per_pt = flow_per_pt[:, :3]
                self.flow_per_pt_r = flow_per_pt[:, 3:]

            # Right now, doing this with pointwise loss.
            flows = []

            # Tracking for consistency loss
            self.means = []
            self.rot_flows = []
            for i in range(len(data.ptr)-1):
                idx1 = data.ptr[i].detach().cpu().numpy().item()
                idx2 = data.ptr[i+1].detach().cpu().numpy().item()
                posi_one = data.pos[idx1:idx2]  # just this PCL's xyz (tool+items).
                tool_one = torch.where(data.x[idx1:idx2, 0] == 1)[0]  # tool 0-th col

                # Just this PCL's flow, but split into translation and rotation.
                flow_one_t = self.flow_per_pt[idx1:idx2]
                flow_one_r = self.flow_per_pt_r[idx1:idx2]

                # Translation is just the average of the _tool_ flow.
                mean_one = torch.mean(flow_one_t[tool_one], dim=0) # (3,)

                # We get rotation by passing tool flow to SVD.
                xyz  = posi_one[tool_one]
                flow = flow_one_r[tool_one]  # note the `r`

                # Get a _transform_ to apply on the original tool pos.
                rot_matrices, _ = flow2pose(
                        xyz=xyz[None,:],
                        flow=flow[None,:],
                        weights=None,
                        return_transform3d=False,
                        return_quaternions=False,
                        world_frameify=False,
                )

                # Correct rotation frame
                xyz_mean = xyz.unsqueeze(0).mean(dim=1, keepdims=True) # (1,1,3)
                t_correction = (xyz_mean - torch.bmm(xyz_mean, rot_matrices)).squeeze(1) # (1,3)

                # Compute a transformation.
                trfm = Rotate(rot_matrices).translate(mean_one[None,:] + t_correction)

                # Applies on ALL points, but later in loss fxn, we filter by tool.
                trfm_xyz = trfm.transform_points(posi_one).squeeze(0)
                batch_flow = trfm_xyz - posi_one
                flows.append(batch_flow)

                # Consistency loss flow computations. Mean is just raw `mean_one`.
                post_flow_t = torch.zeros_like(flow_one_t)
                post_flow_t += mean_one
                self.means.append(post_flow_t)

                # rot_matrices: (1,3,3), t_correction: (1,3), uses the correction so
                # that this rotates w.r.t. the tool origin, producing rotation flow.
                trfm_only_r = Rotate(rot_matrices).translate(t_correction)
                post_flow_r = trfm_only_r.transform_points(posi_one).squeeze(0) - posi_one
                self.rot_flows.append(post_flow_r)

            # Combine consistency loss targets, then return all flow.
            self.means = torch.cat(self.means)
            self.rot_flows = torch.cat(self.rot_flows)
            return torch.cat(flows)

        elif self.encoder_type == 'pointnet_dense_tf_3D_MSE':
            # Extracting 1st item in each output; assume it's the tool point of interest.
            dense_transforms = []
            for i in range(len(data.ptr)-1):
                idx1 = data.ptr[i].detach().cpu().numpy().item()
                dense_transf = flow_per_pt[idx1]
                dense_transforms.append(dense_transf[None,:])
            dense_transforms = torch.cat(dense_transforms)
            return dense_transforms.squeeze()

        elif self.encoder_type == 'pointnet_dense_tf_6D_MSE':
            # Extracting 1st item in each output; assume it's the tool point of interest.
            dense_transforms = []
            for i in range(len(data.ptr)-1):
                idx1 = data.ptr[i].detach().cpu().numpy().item()
                dense_transf = flow_per_pt[idx1]
                dense_transforms.append(dense_transf[None,:])
            dense_transforms = torch.cat(dense_transforms)
            return dense_transforms.squeeze()

        elif self.encoder_type == 'pointnet_dense_tf_6D_pointwise':
            # We apply transform on ALL points and later filter by the tool points.
            # NOTE(daniel): returns flow so that we can do MSE on this for the
            # pointwise loss, but when taking env steps, the actual _action_
            # should be from the single dense transform (1st point).

            # Get tool tip pos for translation correction here.
            if len(info.shape) == 1:
                info = info.unsqueeze(0)
            tool_tip_pos = info[:,:3]

            # The network doesn't predict flow, but we use pointwise matching loss.
            # Given (o(t), a(t)), we apply transform to get a predicted a(t).
            flows = []

            for i in range(len(data.ptr)-1):
                tip_pos_one = tool_tip_pos[i][None, None, :] # (1,1,3)
                idx1 = data.ptr[i].detach().cpu().numpy().item()
                idx2 = data.ptr[i+1].detach().cpu().numpy().item()
                posi_one = data.pos[idx1:idx2]  # just this PCL's xyz (tool+items).
                tool_one = torch.where(data.x[idx1:idx2, 0] == 1)[0]  # tool 0-th col

                # Compute a transformation from just `dense_transf`, no SVD.
                dense_transf = flow_per_pt[idx1]  # just this PCL's _first_ point.
                dense_transf = dense_transf * self._mask  # note the mask!
                mean_one = dense_transf[:3]
                rot_matrices = axis_angle_to_matrix(dense_transf[3:]).transpose(0, 1).unsqueeze(0)

                # Correct rotation frame.
                t_correction = (tip_pos_one - torch.bmm(tip_pos_one, rot_matrices)).squeeze(1) # (1,3)

                # Compute a transformation.
                trfm = Rotate(rot_matrices).translate(mean_one[None,:] + t_correction)

                # Applies on ALL points, but later in loss fxn, we filter by tool.
                trfm_xyz = trfm.transform_points(posi_one).squeeze(0)
                batch_flow = trfm_xyz - posi_one
                flows.append(batch_flow)

            return torch.cat(flows)

        else:
            raise ValueError(self.encoder_type)
