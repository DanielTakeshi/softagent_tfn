# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py
import torch
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius
torch.set_printoptions(linewidth=180, precision=5)
from pytorch3d.transforms import (Rotate, axis_angle_to_matrix)
from rpmg.rpmg import (RPMG, simple_RPMG)
from rpmg.rpmg_losses import rpmg_forward


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2_Class(torch.nn.Module):
    """NOTE(daniel): PointNet++ from the NeurIPS 2017 paper.

    We need to adjust the input dimension to take into account our `data.x`
    which has the features, whereas `data.pos` is known to be 3D. Other than
    those changes, I haven't made changes compared to standard classification
    PointNet++.

    We need to construct `data.x, data.pos, data.batch` for the forward pass.
    For now I do this by constructing `Data` tuples when we have minibatches.
    PointNet++ may require smaller minibatch sizes compared to CNNs. On PyG,
    each PC for the ModelNet10 data is of dim (1024,3), and with a batch size
    of 32 it was taking ~5G of GPU RAM.

    05/19/2022: if we assume geodesic distance, we must normalize output.
    05/27/2022: adding special case of encoder type with classification but
        if we use pointwise loss. We use the per-point flow instead, which
        has the same minimum as the per-point MSE on the future tool.
    05/31/2022: ah, fixed bug: I was not re-scaling the PCL values back to
        the original value, as I was in segm PN++, we should do that for a
        fair comparison with: {class PN++ and pointwise loss}.
    06/02/2022: actually for pointwise baseline we really should remove any
        non-used rotations before we convert to axis-angle. That should help
        with training.
    """

    def __init__(self, in_dim, out_dim, encoder_type, scale_pcl_val=None,
                 normalize_quat=False, n_epochs=None, rpmg_lambda=None, lambda_rot=None):
        super().__init__()
        self.in_dim = in_dim  # 3 for pos, then rest for segmentation
        self.out_dim = out_dim  # the action dimension
        self.encoder_type = encoder_type
        self.scale_pcl_val = scale_pcl_val
        self.normalize_quat = normalize_quat
        self.n_epochs = n_epochs
        self.rpmg_lambda = rpmg_lambda
        self.lambda_rot = lambda_rot
        self.raw_out = None
        self._mask = None

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([self.in_dim, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, out_dim], dropout=0.5, batch_norm=False)

    def assign_clear_mask(self, mask):
        """
        For pouring or scooping, in case we have a 6D vector and need to extract
        a transformation from it here (and want to clear out unused components).
        """
        self._mask = mask

    def forward(self, data, info=None, epoch=None, rgt=None):
        """Standard forward pass, except we might need a custom `data`.

        Just return MLP output, no log softmax's as there's no classification,
        we are either doing regression or segmentation. This is for regression.
        If we are using geodesic distance, we assume last 4 parts form a quat,
        so we normalize the output.
        """

        # Special case if we scaled everything (e.g., meter -> millimeter), need
        # to downscale the observation because otherwise PN++ produces NaNs. :(
        if self.scale_pcl_val is not None:
            data.pos /= self.scale_pcl_val

        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, _, _ = sa3_out
        out = self.mlp(x)

        # Should revert `data.pos` back to the original scale.
        if self.scale_pcl_val is not None:
            data.pos *= self.scale_pcl_val

        if self.normalize_quat:
            # shapes in operation: (batch_size, 4)  / (batch_size, 1)
            assert self.out_dim == 7, self.out_dim
            out_q = out[:,3:] / torch.norm(out[:,3:], dim=1, keepdim=True)
            out = torch.cat([out[:,:3], out_q], dim=1)

        if self.encoder_type == 'pointnet_classif_6D_pointwise':
            # In this special case, we actually want to return flow, which
            # we can compare with ground truth to determine pointwise loss.
            # During inference time we need a special case to just take `out`
            # computed earlier. Similar to 'pointnet_dense_tf_6D_pointwise'
            # but dense_transf is directly the _only_ output of the PN++.
            self.raw_out = out

            # Get rotation center (scooping or pouring) for translation correction.
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

                # Compute a transformation from just `dense_transf`, no SVD.
                dense_transf = self.raw_out[i] * self._mask  # note the mask!
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
        elif self.encoder_type == 'pointnet_rpmg_pointwise':
            # WARNING: DO NOT USE WITH INTRINSIC ROTATIONS!
            out_rot = simple_RPMG.apply(out[:, 3:], 50., self.rpmg_lambda, self.lambda_rot)
            out_rot_raw = out_rot.reshape(-1, 9)
            self.raw_out = torch.cat((out[:, :3], out_rot_raw), dim=1)

            # Get rotation center (scooping or pouring) for translation correction.
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

                # Compute a transformation from just `dense_transf`, no SVD.
                mean_one = out[i, :3]
                rot_matrices = out_rot[i].unsqueeze(0)

                # Correct rotation frame.
                t_correction = (tip_pos_one - torch.bmm(tip_pos_one, rot_matrices)).squeeze(1) # (1,3)

                # Compute a transformation.
                trfm = Rotate(rot_matrices).translate(mean_one[None,:] + t_correction)

                # Applies on ALL points, but later in loss fxn, we filter by tool.
                trfm_xyz = trfm.transform_points(posi_one).squeeze(0)
                batch_flow = trfm_xyz - posi_one
                flows.append(batch_flow)

            self.flow_per_pt = torch.cat(flows)
            self.flow_per_pt_r = None

            return torch.cat(flows)
        elif self.encoder_type == 'pointnet_rpmg':
            # We do the default thing for tau as in the RPMG paper
            # More concretely, we linearly scale tau from tau_init=1/20 to tau_converge=1/4
            # as training progresses. See section 4.3 (in RPMG) and B.1 for a proof of
            # tau_converge=1/4
            if epoch is None:
                tau = 1 / 4
            else:
                tau = 1 / 20 + (1 / 4 - 1 / 20) / 9 * min(epoch // (self.n_epochs // 10), 9)
            out_rot = simple_RPMG.apply(out[:, 3:], tau, self.rpmg_lambda, self.lambda_rot)
            out_rot = out_rot.reshape(-1, 9)

            out = torch.cat((out[:, :3], out_rot), dim=1)
            return out
        elif self.encoder_type == 'pointnet_rpmg_forward':
            # Brian/Chuer stand-alone method method returns (batch,3,3). This is just the
            # forward pass and doesn't produce the proper gradient, use as a baseline only.
            out_rot = rpmg_forward(out[:, 3:])
            out_rot = out_rot.reshape(-1, 9)

            # Concatenate --> (batch, 3+9)
            out = torch.cat((out[:, :3], out_rot), dim=1)
            return out
        elif self.encoder_type == 'pointnet_rpmg_taugt':
            if rgt is not None:
                rgt = rgt.reshape(-1, 12)[:, 3:]
                rgt = rgt.reshape(-1, 3, 3)

            out_rot = simple_RPMG.apply(out[:, 3:], -1, self.rpmg_lambda, self.lambda_rot, rgt)
            out_rot = out_rot.reshape(-1, 9)

            out = torch.cat((out[:, :3], out_rot), dim=1)
            return out
        else:
            return out
