import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle
from bc import utils
from bc.encoder import make_encoder
from bc.se3 import flow2pose
from bc.env_specific_hacks import get_env_action_mask, get_state_dim
import torch_geometric
torch.set_printoptions(edgeitems=10)

# The PointNet++ architectures we experimented with for the CoRL 2022 submission.
from bc.pointnet2_classification import PointNet2_Class
from bc.pointnet2_segmentation import PointNet2_Segm

LOG_FREQ = 10000
PIXEL_ENC = ['pixel', 'segm']
PCL_OBS = ['point_cloud', 'point_cloud_gt_v01', 'point_cloud_gt_v02']
PCL_MODELS = [
    'pointnet', 'pointnet_rpmg', 'pointnet_rpmg_forward', 'pointnet_svd', 'pointnet_avg', 'pointnet_svd_centered',
    'pointnet_svd_pointwise', 'pointnet_svd_pointwise_6d_flow',
    'pointnet_dense_tf_3D_MSE', 'pointnet_dense_tf_6D_MSE', 'pointnet_dense_tf_6D_pointwise',
    'pointnet_classif_6D_pointwise', 'pointnet_svd_6d_flow_mse_loss',
    'pointnet_svd_pointwise_PW_bef_SVD', 'state_predictor_then_mlp', 'pointnet_rpmg_pointwise', 'pointnet_rpmg_taugt'
]
PCL_COMPRESS = [
    'pointnet_svd', 'pointnet_avg', 'pointnet_svd_centered',
    'pointnet_svd_6d_flow_mse_loss', 'pointnet_svd_pointwise_PW_bef_SVD'
]
FLOW_ACTS = ['flow', 'ee2flow', 'ee2flow_sep_rt']
ENV_ACT_MODES = ['translation', 'translation_yrot', 'translation_axis_angle']
AVOID_FLOW_PREDS = [
    'pointnet_classif_6D_pointwise','pointnet_dense_tf_6D_pointwise',
    'pointnet_svd_6d_flow_mse_loss', #'pointnet_rpmg_pointwise'
]
ROT_REPRESENTATIONS = ['rotation_4D', 'rotation_6D', 'rotation_9D', 'rotation_10D',
                       'intrinsic_rotation_4D', 'intrinsic_rotation_6D', 'intrinsic_rotation_9D',
                       'intrinsic_rotation_10D', 'rpmg_flow_6D', 'no_rpmg_6D', 'no_rpmg_9D', 'no_rpmg_10D']
NO_RPMG_REPRESENTATIONS = ['no_rpmg_6D', 'no_rpmg_9D', 'no_rpmg_10D']

# Dont add a model here unless we also plan to test with consistency loss.
POINTWISE_LOSSES = ['pointnet_svd_pointwise', 'pointnet_svd_pointwise_6d_flow']

# The {ee, flow2ee} refer to translation-only EE delta changes.
# Use {eepose, flow2eepose} to refer to translation + rotation delta changes.
ALL_ACTS = ['flow', 'ee2flow', 'ee', 'flow2ee', 'eepose', 'flow2eepose', 'ee2flow_sep_rt', 'eepose_convert']

# Valid (act_type (SoftGym), act_type (BC target), encoder_type) combos.
# Note: `ee` refers to just ee position without any rotations.
# This is only for segmented point cloud inputs!
VALID_COMBOS = [
    ('translation',            'ee',      'pointnet'),     # 'class' PN++, one posi
    ('translation',            'flow2ee', 'pointnet'),     # 'class' PN++, one posi
    ('translation',            'flow',    'pointnet'),     # 'segm' PN++,  full tool flow
    ('translation',            'ee2flow', 'pointnet'),     # 'segm' PN++,  full tool flow
    ('translation',            'flow',    'pointnet_svd_pointwise'),  # 'segm' PN++
    ('translation',            'ee2flow', 'pointnet_svd_pointwise'),  # 'segm' PN++
    ('translation',            'ee',      'pointnet_svd'), # 'segm' PN++, one posi
    ('translation',            'flow2ee', 'pointnet_svd'), # 'segm' PN++, one posi
    ('translation',            'ee',      'pointnet_svd_centered'), # 'segm' PN++, one posi
    ('translation',            'flow2ee', 'pointnet_svd_centered'), # 'segm' PN++, one posi
    ('translation',            'ee',      'pointnet_avg'), # 'segm' PN++, one posi
    ('translation',            'flow2ee', 'pointnet_avg'), # 'segm' PN++, one posi
    ('translation_yrot',       'eepose',  'pointnet'),     # 'class' PN++, one pose
    ('translation_yrot',       'eepose',  'pointnet_svd'), # 'segm' PN++, one pose
    ('translation_axis_angle', 'eepose',  'pointnet'),     # 'class' PN++, one pose
    ('translation_axis_angle', 'eepose',  'pointnet_svd'), # 'segm' PN++, one pose
    ('translation_axis_angle', 'flow',    'pointnet_svd_pointwise'),  # 'segm' PN++
    ('translation_axis_angle', 'flow',    'pointnet'),  # 'segm' PN++ per-point matching
    ('translation_axis_angle', 'ee2flow', 'pointnet_svd_pointwise'),
    ('translation_axis_angle', 'ee2flow', 'pointnet'),
    ('translation',            'flow',    'pointnet_svd_pointwise_6d_flow'),  # 'segm' PN++, 6D flow
    ('translation_axis_angle', 'ee2flow', 'pointnet_svd_pointwise_6d_flow'),  # 'segm' PN++, 6D flow
    ('translation',            'ee',      'pointnet_dense_tf_3D_MSE'),          # dense tf
    ('translation_axis_angle', 'eepose',  'pointnet_dense_tf_6D_MSE'),          # dense tf
    ('translation_axis_angle', 'ee2flow', 'pointnet_dense_tf_6D_pointwise'),    # dense tf
    ('translation_axis_angle', 'ee2flow', 'pointnet_classif_6D_pointwise'),     # pw baseline ablation
    ('translation_axis_angle', 'ee2flow', 'pointnet_svd_pointwise_PW_bef_SVD'),     # pw baseline ablation
    ('translation_axis_angle', 'eepose',  'pointnet_svd_6d_flow_mse_loss'),     # ablation
    ('translation_axis_angle', 'ee2flow_sep_rt', 'pointnet_svd_pointwise_6d_flow'), # ablation
    ('translation_axis_angle', 'eepose',  'state_predictor_then_mlp'),     # rebuttal
    # [eepose] with rotation conversion code to test different rotation representations
    ('translation_yrot',       'eepose_convert',  'pointnet'),     # 'class' PN++, one pose
    ('translation_yrot',       'eepose_convert',  'pointnet_svd'), # 'segm' PN++, one pose
    ('translation_axis_angle', 'eepose_convert',  'pointnet'),     # 'class' PN++, one pose
    ('translation_axis_angle', 'eepose_convert',  'pointnet_svd'), # 'segm' PN++, one pose
    ('translation_axis_angle', 'eepose_convert',  'pointnet_dense_tf_6D_MSE'),          # dense tf
    ('translation_axis_angle', 'eepose_convert',  'pointnet_svd_6d_flow_mse_loss'),     # ablation
    ('translation_axis_angle', 'eepose_convert',  'state_predictor_then_mlp'),     # rebuttal
    ('translation_axis_angle', 'eepose_convert',  'pointnet_rpmg'),     # 'class' PN++, RPMG
    ('translation_axis_angle', 'eepose_convert',  'pointnet_rpmg_forward'),     # 'class' PN++, RPMG forward only
    ('translation_axis_angle', 'eepose_convert',  'pointnet_rpmg_taugt'),     # 'class' PN++, RPMG tau_gt
    ('translation_axis_angle', 'ee2flow', 'pointnet_rpmg_pointwise'), # 'class' PN++, RPMG, pointwise loss
]


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """Policy network for Behavioral Cloning."""

    def __init__(self, args, device, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, act_type):
        """This uses an encoder class across all types of encoder types.

        The main benefit is to fix comparisons across different types of input
        observations and to compare CNNs vs MLP, etc. With PointNet++, we don't
        really need this so (as with MLPs) we make the encoder the identity.
        Actually I think it's standard usage to treat the 'encoder' of PointNet++
        as any stuff before the final set of MLPs ...
        """
        super().__init__()
        self.args = args
        self.device = device
        self.outputs = dict()
        self.encoder_type = encoder_type

        # This mainly changes the computation for the forward pass where we don't
        # divide the first 3 channels by 255. Used for those with depth or segmentation
        # except for RGBD (due to the order we implemented this...). TODO change name.
        use_depth_segm = args.env_kwargs['observation_mode'] in ['depth_segm', 'depth_img',
                'rgb_segm_masks', 'rgbd_segm_masks']

        encoder_obs_shape = obs_shape

        # In this case obs_shape is (2000,d) but we want the _state_ obs! This is because
        # we still get the point_cloud obs type, but we pass that through a separate net.
        if encoder_type == 'state_predictor_then_mlp':
            state_dim = get_state_dim(self.args.env_name)
            encoder_obs_shape = (state_dim,)

        # Mainly for the CNN; PN++ variants set this as the identity. (And the MLP)
        self.encoder = make_encoder(
            encoder_type, encoder_obs_shape, encoder_feature_dim, num_layers, num_filters,
            depth_segm=use_depth_segm,
        )

        # Use 'Segm' network if: (a) gt actions are flow, or (b) we're using a
        # PointNet segm variant which compresses flow into single poses, or (c)
        # a dense transformation policy.
        if encoder_type == 'state_predictor_then_mlp':
            # Ack another hack we have to add another note about the state encoder.
            # The issue is that this is also a 'PCL_MODELS'...
            self.trunk = nn.Sequential(
                nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_shape[0])
            )
            self.apply(weight_init)  # prob not for PointNet?
        elif encoder_type in PCL_MODELS:
            in_dim = obs_shape[1]

            if (((act_type in FLOW_ACTS) or (encoder_type in PCL_COMPRESS) or self.args.dense_transform)
                    and encoder_type not in ['pointnet_classif_6D_pointwise', 'pointnet_rpmg_pointwise']):
                # In two cases (thus far) we'd like the net to output 6D stuff.
                # Misleading name since dense transf policy doesn't output flow. :-(
                flow_dim = 3
                if ('6d_flow' in encoder_type) or ('dense_tf_6D_' in encoder_type):
                    flow_dim = 6

                # Form the 'trunk'. This is really the 'encoder', bad terminology.
                self.trunk = PointNet2_Segm(
                        in_dim=in_dim,
                        flow_dim=flow_dim,
                        encoder_type=encoder_type,
                        scale_pcl_val=self.args.scale_pcl_val,
                        separate_MLPs_R_t=self.args.separate_MLPs_R_t,
                        dense_transform=self.args.dense_transform,
                        remove_skip_connections=self.args.remove_skip_connections,
                )
            else:
                # Use the classification based version of PointNet++.
                # Also modify this for the different rotation representations.
                out_dim = self._get_out_dim(args, action_shape)
                self.trunk = PointNet2_Class(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        encoder_type=encoder_type,
                        scale_pcl_val=self.args.scale_pcl_val,
                        n_epochs=self.args.n_epochs,
                        rpmg_lambda=self.args.rpmg_lambda,
                        lambda_rot=self.args.lambda_rot,
                )

            # Clears out unused DoFs in the action space.
            mask = get_env_action_mask(
                    env_name=args.env_name,
                    env_version=args.env_version,
                    device=self.device
            )
            self.trunk.assign_clear_mask(mask)
        else:
            self.trunk = nn.Sequential(
                nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_shape[0])
            )
            self.apply(weight_init)  # prob not for PointNet?

    def forward(self, obs, info=None, epoch=None, rgt=None):
        obs = self.encoder(obs)
        if self.encoder_type in PCL_MODELS and self.encoder_type != 'state_predictor_then_mlp':
            act = self.trunk(obs, info=info, epoch=epoch, rgt=rgt)
        else:
            act = self.trunk(obs)
        return act

    def _get_out_dim(self, args, action_shape):
        """Handle the output dim for the classification PN++.

        Output dim is 3 (for translation) plus X (for rot representation).
        NOTE(daniel): geodesic dist is deprecated and not suported.
        """
        out_dim = action_shape[0]
        if args.rotation_representation in ROT_REPRESENTATIONS:
            if '4D' in args.rotation_representation:
                out_dim = 3 + 4
            elif '6D' in args.rotation_representation:
                out_dim = 3 + 6
            elif '9D' in args.rotation_representation:
                out_dim = 3 + 9
            elif '10D' in args.rotation_representation:
                out_dim = 3 + 10
        if args.use_geodesic_dist:
            raise NotImplementedError()
        return out_dim


class BCAgent(object):
    """Behavioral Cloning (BC) Agent."""

    def __init__(self,
                 obs_shape,
                 action_shape,
                 device,
                 args,
                 hidden_dim=256,
                 actor_lr=1e-3,
                 encoder_type='pixel',
                 encoder_feature_dim=50,
                 num_layers=4,
                 num_filters=32,
                 log_interval=100):
        self.args = args
        self.device = device
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.encoder_type = encoder_type
        self.act_type = args.act_type

        self.actor = Actor(args, device,
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, self.act_type,
        ).to(device)

        # Use nn.MSELoss() both for regressing to one action and for tool flow pred.
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.MSE_loss = nn.MSELoss()

        # In case we use weighted loss, should be for baseline methods only.
        self.use_geodesic_dist = args.use_geodesic_dist
        self.use_weighted_R_t_loss = args.weighted_R_t_loss
        if args.rotation_representation in NO_RPMG_REPRESENTATIONS:
            if '4D' in args.rotation_representation:
                n_weights = 3 + 4
            elif '6D' in args.rotation_representation:
                n_weights = 3 + 6
            elif '9D' in args.rotation_representation:
                n_weights = 3 + 9
            elif '10D' in args.rotation_representation:
                n_weights = 3 + 10
        elif args.rotation_representation in ROT_REPRESENTATIONS:
            n_weights = 12  # 3 for translation, 9 for rotation
        else:
            n_weights = action_shape[0]
        self.weights = torch.ones(n_weights).to(self.device)
        self.weights[:3] = args.lambda_pos

        # We handle rotation weighting in the RPMG backward pass
        # by scaling the gradients, as doing this in our loss
        # messes with the Riemannian gradient
        if 'pointnet_rpmg' in encoder_type:
            self.weights[3:] = 1.0
        else:
            self.weights[3:] = args.lambda_rot

        self.lambda_pos = args.lambda_pos
        self.lambda_rot = args.lambda_rot

        # Extra losses.
        self.use_consistency_loss = args.use_consistency_loss
        self.use_dense_loss = args.use_dense_loss
        self.lambda_consistency = args.lambda_consistency
        self.lambda_dense = args.lambda_dense

        # We use `utils.eval_mode()` for test-time evaluation.
        if args.load_model:
            self.load(args.load_model_path)
        self.train()
        self._debug()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def set_state_encoder(self, pcl_to_state_encoder):
        """Set the state encoder here, might be easier.

        This should be trained ahead of time and then fixed. So it is not part
        of `self.actor` and should therefore not get optimized.
        """
        self.pcl_to_state_encoder = pcl_to_state_encoder

    def weighted_R_t_loss(self, input, target):
        """Use weights for separate rotation and translation losses.

        Should be for baseline methods to show how difficult it is to tune such
        objectives. for rotations, we can use both MSEs (for axis-angles) or
        geodesic distances (for quaternions). The latter uses: 1 - <q1,q2>^2.
        https://math.stackexchange.com/questions/90081/quaternion-distance
        Note that this requires quaternions to be normalized. The targets are
        normalized via `axis_angle_to_quaternion` (empirically) while the
        inputs are from the network, where we forcibly add normalization.

        input, target: (N,6) or (N,7) arrays with first 3 columns indicating
        the translation of the policy versus ground truth, respectively.
        """
        if self.use_geodesic_dist:
            raise ValueError()
            out_t = self.lambda_pos * (input[:,:3] - target[:,:3])**2
            out_R = self.lambda_rot * (1 - torch.sum(input[:,3:] * target[:,3:], dim=1)**2)
            weighted = 0.5 * (out_t.mean() + out_R.mean())
        elif self.args.rotation_representation in ROT_REPRESENTATIONS and self.args.rotation_representation not in NO_RPMG_REPRESENTATIONS:
            # This is assuming a Frobenius norm loss on rotations, might want others?
            out = (input - target) ** 2
            rotation_loss = out[:, 3:].sum()
            translation_loss = out[:, :3].mean()
            weighted = self.lambda_pos * translation_loss + rotation_loss
            # weighted = (self.weights * out).mean()
        else:
            out = (input - target) ** 2
            weighted = (self.weights * out).mean()
        return weighted

    def get_single_flow(self, obs, info=None, act_gt=None):
        """
        Given obs, gets the flow of the first observation in the batch for debugging.
        If scaling, then `obs` will be scaled (e.g., in mm units) but in the forward
        pass we divide back into meters to avoid NaNs. But we will end up re-scaling
        it back into the original values (so use `obs.pos` directly).

        Called both in minibatch and non-minibatch cases.
        """
        assert self.act_type in FLOW_ACTS or self.encoder_type in PCL_COMPRESS
        if not isinstance(obs, torch_geometric.data.Data):
            obs = utils.create_pointnet_pyg_data(obs, self.device)
        if info is not None and not isinstance(info, torch.Tensor):
            info = torch.from_numpy(info)
            info = info.to(torch.float32).to(self.device)

        # Forward pass, assigns flow to `actor.trunk.flow_per_pt`
        act = self.actor(obs, info=info)
        pts_in_pcl = np.sum(obs.batch.cpu().numpy() == 0)
        flow_1 = self.actor.trunk.flow_per_pt[:pts_in_pcl]
        tool_1 = obs.x[:pts_in_pcl]
        pos_1 = obs.pos[:pts_in_pcl]

        tool_one = torch.where(tool_1[:,0] == 1)[0]
        xyz = pos_1[tool_one]
        flow = flow_1[tool_one]

        ret_dict = {
            'xyz': xyz,
            'flow': flow,
        }

        if act_gt is not None:
            # First two cases apply if using PN++ avg where each mb item has 3-dim.
            if act_gt.shape == torch.Size([self.args.batch_size, 3]):
                act_gt = act_gt[0]
                return xyz, flow, act_gt.expand(xyz.shape[0], 3)
            elif act_gt.shape == torch.Size([3]):
                return xyz, flow, act_gt.expand(xyz.shape[0], 3)
            else:
                # act_gt is (N,flow_dim), just get those from the first one.
                act_gt = act_gt[:pts_in_pcl]

            ret_dict['act_gt'] = act_gt[tool_one]
            # return xyz, flow, act_gt[tool_one]

        # Extract actual reported flow if possible
        if self.act_type in FLOW_ACTS:
            ret_dict['act'] = act[tool_one]

        # Extract rotation flow if possible
        if self.actor.trunk.flow_per_pt_r is not None:
            ret_dict['flow_r'] = self.actor.trunk.flow_per_pt_r[tool_one]

        # return xyz, flow
        return ret_dict

    def select_action(self, obs, info=None):
        """Select action, used for BC only to evaluate.

        We should return an action that we can use for the env, modulo processing
        only to scale action magnitudes and to 'denormalize'. Thus, should handle
        the flow -> action pipeline (if desired).

        Also, even if we are scaling the flow, the input should still be in the
        raw coordinates (meters) since too large an input leads to NaNs.
        """
        with torch.no_grad():
            if self.encoder_type == 'state_predictor_then_mlp':
                # New state encoder. This should turn (2000,d) PCL to (1,state_dim) torch.
                # Minibatch size of 1 since this is just a single test-time action selection.
                obs_pn2 = utils.create_pointnet_pyg_data(obs, self.device)
                obs = self.pcl_to_state_encoder(obs_pn2)
                if not isinstance(obs, torch.Tensor):
                    obs = torch.from_numpy(obs)
                obs = obs.to(torch.float32).to(self.device)
                act = self.actor(obs, info=info)  # then (1,state_dim) to (1,action)
            elif self.encoder_type in PCL_MODELS:
                obs_pn2 = utils.create_pointnet_pyg_data(obs, self.device)
                if info is not None and not isinstance(info, torch.Tensor):
                    info = torch.from_numpy(info)
                    info = info.to(torch.float32).to(self.device)
                act = self.actor(obs_pn2, info=info)

                # NOTE (eddie): We run _flow_to_env_action for all segmentation outputs even if it's a PCL_COMPRESS model. This is because pointnet_svd_centered outputs actions in the wrong frame
                # However, sometimes it might be desirable to take the PCL_COMPRESS outputs. If we'd like to do this, we should implement some flag for it.
                if self.encoder_type == 'pointnet_classif_6D_pointwise':
                    # Extract the action (transl,axis_angle) directly from output.
                    act = self.actor.trunk.raw_out[0]  # shape (6)
                elif self.encoder_type == 'pointnet_rpmg_pointwise':
                    act = self.actor.trunk.raw_out[0] # shape (3 + 9,)
                elif self.encoder_type == 'pointnet_dense_tf_6D_pointwise':
                    # Extract the action (transl,axis_angle) from index 0 in the PCL.
                    # This is not 'flow' but a dense transform.
                    act = self.actor.trunk.flow_per_pt[0]  # shape (6)
                elif self.encoder_type in ['pointnet_svd_6d_flow_mse_loss', 'pointnet_svd']:
                    # We could do flow2act but this will already produce a desired
                    # transformation with `act`, and should be the same as what
                    # `flow_to_env_act` gives with `svd` (I've empirically tested).
                    pass
                elif (self.act_type in FLOW_ACTS or self.encoder_type in PCL_COMPRESS):
                    # kind of a hack, we pull flow_per_pt out of self.actor.trunk
                    # If we scaled flow, this produces (R,t) with scaled t.
                    act = self._flow_to_env_action(
                            obs=obs_pn2,
                            flow=self.actor.trunk.flow_per_pt,
                            flow_r=self.actor.trunk.flow_per_pt_r,
                            info=info)
            else:
                if not isinstance(obs, torch.Tensor):
                    obs = torch.from_numpy(obs)
                if info is not None and not isinstance(info, torch.Tensor):
                    info = torch.from_numpy(info)
                    info = info.to(torch.float32).to(self.device)
                obs = obs.to(torch.float32).to(self.device)
                obs = obs.unsqueeze(0)
                act = self.actor(obs, info=info)
            return act.cpu().data.numpy().flatten()

    def update(self, replay_buffer, epoch=None):
        """Just updates the actor (policy). There's no critic.

        Using MSE since we have continuous actions. The `act_pol` and `act_gt`
        have shapes (batch_size, act_dim) each. Take component-wise difference,
        find L2 diff across 'rows', then average. Only consider tool points!
        Actually shapes can be different if using separate (r,t) as the action
        type for pointwise 6d flow supervision (the 'sep_rt' case).

        Apply a tool mask if using tool flow as the 'action' since the segmented
        point cloud has rows in its array corresponding to non-tool points, and we
        don't do MSE on those. Used in pointwise, consistency, and dense losses.
        In this case, `act_gt` & `act_pol` encode per-point flow of some sort.

        Separate MSEs for pos and rot are solely for debugging.
        """
        obs, act_gt, info = replay_buffer.sample_obs_act(get_info=True)

        if self.encoder_type == 'state_predictor_then_mlp':
            with torch.no_grad():
                # First case, handle this first, turn minibatch of PCLs to a minibatch
                # of state tensors. Then pass to the actual actor. I think we need the
                # torch.no_grad() to use the state encoder as a fixed feature extractor?
                obs = self.pcl_to_state_encoder(obs)

        # Get output, mask out tool (in both prediction & target) if needed.
        act_pol = self.actor(obs, info=info, epoch=epoch, rgt=act_gt)

        # TODO(eddie): make ablation special case less special
        if (self.act_type in FLOW_ACTS and 'sep_rt' not in self.act_type):
            act_pol, act_gt = self._mask_out_tools(obs, act_pol, act_gt)

        # Compute either MSE on action vectors or pointwise loss.
        if self.use_weighted_R_t_loss:
            actor_loss = self.weighted_R_t_loss(act_pol, act_gt)
        elif self.encoder_type == 'pointnet_svd_pointwise_PW_bef_SVD':
            # Pointwise loss before SVD (3D flow). Want predicted flow vectors from
            # the trunk, and then compare that with the direct act_gt (with flows).
            act_3d = self.actor.trunk.flow_per_pt
            act_3d, act_gt = self._mask_out_tools(obs, act_3d, act_gt)
            actor_loss = self.MSE_loss(act_3d, act_gt)
        elif 'sep_rt' in self.act_type and (not self.use_dense_loss):
            # Pointwise loss before SVD (6D flow). Also if using dense loss, but
            # assume this case here is only if we do not also use pointwise loss.
            act_6d = torch.cat((
                self.actor.trunk.flow_per_pt,
                self.actor.trunk.flow_per_pt_r), dim=1)
            act_6d, act_gt = self._mask_out_tools(obs, act_6d, act_gt)
            actor_loss = self.MSE_loss(act_6d, act_gt)
        elif 'sep_rt' in self.act_type and self.use_dense_loss:
            # Use dense loss LATER, apply pointwise loss NOW.
            assert self.actor.trunk.means is not None and act_gt.shape[1] == 6, act_gt.shape
            act_gt_3d = act_gt[:, :3] + act_gt[:, 3:]  # 6d -> 3d
            act_pol, act_gt_3d = self._mask_out_tools(obs, act_pol, act_gt_3d)
            actor_loss = self.MSE_loss(act_pol, act_gt_3d)
        else:
            # Pointwise loss after SVD.
            actor_loss = self.MSE_loss(act_pol, act_gt)

        # Compute consistency loss, does NOT use any ground truth flow info.
        if self.use_consistency_loss:
            assert (self.encoder_type in POINTWISE_LOSSES or
                    self.encoder_type == 'pointnet_svd_6d_flow_mse_loss' or
                    self.encoder_type == 'pointnet_svd')
            if self.actor.trunk.means is None and self.encoder_type == 'pointnet_svd':
                # 3D flow case but with act_pol not the flow (so we have to extract).
                act_raw_flow = self.actor.trunk.flow_per_pt
                flow_target = self.actor.trunk.flows_3d
                act_raw_flow, flow_target = self._mask_out_tools(obs, act_raw_flow, flow_target)
                consistency_loss = self.MSE_loss(act_raw_flow, flow_target)
                actor_loss += self.lambda_consistency * consistency_loss
            elif self.actor.trunk.means is None:
                # 3D flow case
                act_raw_flow = self.actor.trunk.flow_per_pt
                act_raw_flow, act_pol = self._mask_out_tools(obs, act_raw_flow, act_pol)
                consistency_loss = self.MSE_loss(act_raw_flow, act_pol)
                actor_loss += self.lambda_consistency * consistency_loss
            else:
                # 6D flow case
                trans_pred = self.actor.trunk.flow_per_pt
                trans_target = self.actor.trunk.means
                trans_pred, trans_target = self._mask_out_tools(obs, trans_pred, trans_target)
                consistency_loss = self.MSE_loss(trans_pred, trans_target) * 0.5

                rot_pred = self.actor.trunk.flow_per_pt_r
                rot_target = self.actor.trunk.rot_flows
                rot_pred, rot_target = self._mask_out_tools(obs, rot_pred, rot_target)
                consistency_loss += self.MSE_loss(rot_pred, rot_target) * 0.5

                actor_loss += self.lambda_consistency * consistency_loss
        else:
            consistency_loss = torch.tensor([0.0])

        # Dense loss, same as consistency but with g.t. instead of predicted flow.
        if self.use_dense_loss:
            assert (self.encoder_type in POINTWISE_LOSSES or
                    self.encoder_type == 'pointnet_svd_6d_flow_mse_loss')
            if self.actor.trunk.means is None:
                # 3D flow case. NOTE: potentialy some repeated took mask computation.
                act_raw_flow = self.actor.trunk.flow_per_pt
                act_raw_flow, act_pol = self._mask_out_tools(obs, act_raw_flow, act_pol)
                dense_loss = self.MSE_loss(act_raw_flow, act_gt)
                actor_loss += self.lambda_dense * dense_loss
            else:
                # 6D flow case, basically the same as pointwise before SVD.
                act_6d = torch.cat((
                    self.actor.trunk.flow_per_pt,
                    self.actor.trunk.flow_per_pt_r), dim=1)
                act_6d, act_gt = self._mask_out_tools(obs, act_6d, act_gt)
                dense_loss = self.MSE_loss(act_6d, act_gt)
                actor_loss += self.lambda_dense * dense_loss
        else:
            dense_loss = torch.tensor([0.0])

        # optimize the actor, return loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Get MSEs for each separate component, ignoring weights.
        with torch.no_grad():
            pos_loss = self.MSE_loss(act_pol[:,:3], act_gt[:,:3])
            if act_pol.shape[1] > 3:
                rot_loss = self.MSE_loss(act_pol[:,3:], act_gt[:,3:])
            else:
                rot_loss = torch.zeros(1)
        return (actor_loss.item(),
                pos_loss.item(),
                rot_loss.item(),
                consistency_loss.item(),
                dense_loss.item())

    def evaluate_mse(self, obs, act_gt, info=None):
        """For validation, serves as another way to measure BC.

        For PointNet++, again we assume that all have same # of points.
        As with training, only consider the tool points! The separate MSEs
        at the end for pos and rot are for debugging, not actual training.

        NOTE(daniel): this doesn't take into account dense loss, etc.
        """
        with torch.no_grad():
            # First case, handle this first, turn minibatch of PCLs to a minibatch
            # of state tensors. Then pass to the actual actor.
            if self.encoder_type == 'state_predictor_then_mlp':
                obs = self.pcl_to_state_encoder(obs)

            act_pol = self.actor(obs, info=info)
            # TODO(eddie): see comment in update()
            if self.act_type in FLOW_ACTS and 'sep_rt' not in self.act_type:
                act_pol, act_gt = self._mask_out_tools(obs, act_pol, act_gt)

            if self.use_weighted_R_t_loss:
                valid_loss = self.weighted_R_t_loss(act_pol, act_gt)
            elif self.encoder_type == 'pointnet_svd_pointwise_PW_bef_SVD':
                # Pointwise loss before SVD, 3D flow.
                act_3d = self.actor.trunk.flow_per_pt
                act_3d, act_gt = self._mask_out_tools(obs, act_3d, act_gt)
                valid_loss = self.MSE_loss(act_3d, act_gt)
            elif 'sep_rt' in self.act_type:
                # Pointwise loss before SVD, 6D flow.
                act_6d = torch.cat((
                    self.actor.trunk.flow_per_pt,
                    self.actor.trunk.flow_per_pt_r), dim=1)
                act_6d, act_gt = self._mask_out_tools(obs, act_6d, act_gt)
                valid_loss = self.MSE_loss(act_6d, act_gt)
            else:
                valid_loss = self.MSE_loss(act_pol, act_gt)

            # Get MSEs for each separate component, ignoring weights.
            pos_loss = self.MSE_loss(act_pol[:,:3], act_gt[:,:3])
            if act_pol.shape[1] > 3:
                rot_loss = self.MSE_loss(act_pol[:,3:], act_gt[:,3:])
            else:
                rot_loss = torch.zeros(1)
            return (valid_loss.item(), pos_loss.item(), rot_loss.item())

    def save(self, model_dir, epoch):
        """Save model, maybe for pre-training RL later."""
        ckpt = {
            'actor_sd': self.actor.state_dict(),
            'actor_optim_sd': self.actor_optimizer.state_dict(),
        }
        PATH = os.path.join(model_dir, f'ckpt_{str(epoch).zfill(4)}.tar')
        torch.save(ckpt, PATH)
        print(f'BCAgent.save(): {PATH}.')

    def load(self, model_dir):
        """Load model, maybe for RL or just inspecting output.

        We only need the optimizer if we plan to resume training, otherwise just
        the actor_sd seems to be enough.
        """
        ckpt = torch.load(model_dir)
        print(f'BCAgent.load(): {model_dir}.')
        self.actor.load_state_dict(ckpt['actor_sd'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optim_sd'])

    def _debug(self):
        print('\nCreated BC agent. Actor:')
        print(self.actor)
        print(f'parameters: {utils.count_parameters(self.actor)}\n')

    def _mask_out_tools(self, obs, act_pol, act_gt):
        """Only consider the tool points.

        Obs and act arrays should be aligned; mask out based on if the obs
        point cloud has a tool at each point, and do this to act_{pol,gt}.
        We do something similar in PN++ if encoder is in PCL_COMPRESS.
        """
        tool_mask = torch.zeros_like(act_gt).to(self.device)
        tool_pts = torch.where(obs.x[:,0] == 1)[0]  # 0 = tool column
        tool_mask[tool_pts] = 1.  # will create rows of 1s
        act_pol = act_pol * tool_mask
        act_gt = act_gt * tool_mask
        return (act_pol, act_gt)

    def _flow_to_env_action(self, obs, flow, flow_r=None, info=None):
        """Given the flow, must determine env action.

        If translation-only just take average of all tool flow predictions. We
        can extract indices corresponding to the tool from the original obs.
        NOTE! This should only be used for flow actions and where we actually
        need the full flow vector. It would be easier if the PointNet++ variant
        we use compresses the flow into a single pose.

        NOTE! Assumes a single batch item here, not multi-batch.
        We are also calling this even with pointnet_avg as the model, but in
        that case it should be equivalent as we are still taking the tool idxs,
        and averaging the corresponding (tool point) flow.
        """
        assert self.args.method_flow2act in ('svd', 'mean')
        # assert self.encoder_type not in PCL_COMPRESS, self.encoder_type
        assert obs.x.shape[0] == flow.shape[0], f'{obs.x.shape}, {flow.shape}'

        if self.args.method_flow2act == 'mean':
            obs_tool_idxs = torch.where(obs.x[:,0] == 1)[0]
            flow_tool_pts = flow[obs_tool_idxs]
            act = torch.mean(flow_tool_pts, dim=0)
            return act

        elif self.args.method_flow2act == 'svd':
            # I think this code will work for PointWater as well, since `info`
            # is passed as the center of the rotation frame (bottom floor center)
            # even though it's misleadingly named 'tip_pos_one'.
            if len(info.shape) > 1:
                info = info.squeeze(0)
            tip_pos_one = info[:3]

            if flow_r is None:
                # 3D flow case
                tool_one = torch.where(obs.x[:,0] == 1)[0]
                xyz = obs.pos[tool_one]
                flow_t = flow[tool_one]

                rot_matrices, trans = flow2pose(
                    xyz=xyz[None,:],
                    flow=flow_t[None,:],
                    weights=None,
                    return_transform3d=False,
                    return_quaternions=False,
                    world_frameify=False,
                )

                # NOTE(daniel): I think we now want axis angle.
                quats = matrix_to_quaternion(matrix=rot_matrices.transpose(1, 2))
                axis_ang = quaternion_to_axis_angle(quaternions=quats)

                # Correct translation w.r.t. tool origin center.
                xyz_mean = xyz.unsqueeze(0).mean(dim=1, keepdims=True)
                relative_xyz_mean = xyz_mean - tip_pos_one[None, None, :]
                t_correction = relative_xyz_mean - torch.bmm(relative_xyz_mean, rot_matrices)
                trans += t_correction.squeeze(1)

                pred_eepose = torch.cat([trans, axis_ang], dim=1)
                return pred_eepose.squeeze(0)
            else:
                # 6D flow case
                tool_one = torch.where(obs.x[:, 0] == 1)[0]

                # Translation flow
                mean_one = torch.mean(flow[tool_one], dim=0, keepdims=True)

                # Rotation flow
                xyz = obs.pos[tool_one]
                filt_flow_r = flow_r[tool_one]

                rot_matrices, _ = flow2pose(
                    xyz=xyz[None,:],
                    flow=filt_flow_r[None,:],
                    weights=None,
                    return_transform3d=False,
                    return_quaternions=False,
                    world_frameify=False,
                )

                # Convert rotation into axis-angle with transpose!
                quats = matrix_to_quaternion(matrix=rot_matrices.transpose(1, 2))
                axis_ang = quaternion_to_axis_angle(quaternions=quats)

                # Change rotation frame to be around tool origin.
                xyz_mean = xyz.unsqueeze(0).mean(dim=1, keepdims=True)
                relative_xyz_mean = xyz_mean - tip_pos_one[None, None, :]
                t_correction = relative_xyz_mean - torch.bmm(relative_xyz_mean, rot_matrices)
                mean_one += t_correction.squeeze(1)

                pred_eepose = torch.cat([mean_one, axis_ang], dim=1)
                return pred_eepose.squeeze(0)
        else:
            raise ValueError(self.args.method_flow2act)
