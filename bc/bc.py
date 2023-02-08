import os
from tkinter import N
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle
from bc import utils
from bc.encoder import make_encoder
from bc.se3 import flow2pose
import torch_geometric
from collections import defaultdict
import pickle

# For regression and flow, but based on classification and segmentation.
from bc.pointnet2_classification import PointNet2_Regr
from bc.pointnet2_segmentation import PointNet2_Flow

LOG_FREQ = 10000
PIXEL_ENC = ['pixel', 'segm']
PCL_MODELS = ['pointnet', 'pointnet_svd', 'pointnet_avg', 'pointnet_svd_centered',
        'pointnet_svd_pointwise', 'pointnet_svd_pointwise_6d_flow']
PCL_COMPRESS = ['pointnet_svd', 'pointnet_avg', 'pointnet_svd_centered']
FLOW_ACTS = ['flow', 'ee2flow']
ENV_ACT_MODES = ['translation', 'translation_yrot', 'translation_axis_angle']
POINTWISE_LOSSES = ['pointnet_svd_pointwise', 'pointnet_svd_pointwise_6d_flow']

# The {ee, flow2ee} refer to translation-only EE delta changes.
# Use {eepose, flow2eepose} to refer to translation + rotation delta changes.
ALL_ACTS = ['flow', 'ee2flow', 'ee', 'flow2ee', 'eepose', 'flow2eepose']

# Valid (act_mode (SoftGym), act_type (BC target), encoder_type) combos.
# Note: `ee` refers to just ee position without any rotations.
# This is only for segmented point cloud inputs!
#! Adding the new act_mode (translation_axis_angle), act_type (flow), and pointnet svd 6d flow
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
    ('translation', 'ee', 'pointnet_avg'),
    ('translation',            'flow',    'pointnet_svd_pointwise_6d_flow'),  # 'segm' PN++, 6D flow
    ('translation_axis_angle', 'ee2flow',  'pointnet_svd_pointwise_6d_flow'),  # 'segm' PN++, 6D flow
    ('translation_axis_angle', 'flow',  'pointnet_svd_pointwise_6d_flow'),  # 'segm' PN++, 6D flow
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

    def __init__(self, args, obs_shape, action_shape, hidden_dim, encoder_type,
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
        self.outputs = dict()
        self.encoder_type = encoder_type

        # Mainly for the CNN; PN++ variants set this as the identity.
        print('[SARTHAK] obs_shape: {}'.format(obs_shape))
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters
        )

        # Use 'Flow' network if: (a) gt actions are flow, or (b) we're using a
        # PointNet segm variant which compresses flow into single poses, or (c)
        # a dense transformation policy.
        if encoder_type in PCL_MODELS:
            in_dim = obs_shape[1]
            if (act_type in FLOW_ACTS) or (encoder_type in PCL_COMPRESS):
                self.trunk = PointNet2_Flow(
                        in_dim=in_dim,
                        encoder_type=encoder_type,
                        scale_pcl_val=self.args.scale_pcl_val,
                )
            else:
                out_dim = action_shape[0]
                self.trunk = PointNet2_Regr(in_dim=in_dim, out_dim=out_dim)
        else:
            self.trunk = nn.Sequential(
                nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, action_shape[0])
            )
            self.apply(weight_init)  # prob not for PointNet?

    def forward(self, obs, info=None):
        obs = self.encoder(obs)
        if self.encoder_type in PCL_MODELS:
            act = self.trunk(obs, info=info)
        else:
            act = self.trunk(obs)
        return act


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
        self.data_augm_PCL = args.data_augm_PCL

        self.actor = Actor(args,
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, self.act_type,
        ).to(device)

        # Use nn.MSELoss() both for regressing to one action and for tool flow pred.
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.MSE_loss = nn.MSELoss()

        # In case we use weighted loss, should be for baseline methods only.
        self.use_weighted_R_t_loss = args.weighted_R_t_loss
        self.weights = torch.ones(action_shape[0]).to(self.device)
        self.weights[:3] = args.lambda_pos
        self.weights[3:] = args.lambda_rot
        self.lambda_pos = args.lambda_pos
        self.lambda_rot = args.lambda_rot

        # Consistency loss arguments
        self.use_consistency_loss = args.use_consistency_loss
        self.lambda_consistency = args.lambda_consistency

        # We use `utils.eval_mode()` for test-time evaluation.
        if args.load_model:
            self.load(args.load_model_path)
        else:
            self.train()
        self._debug()
        self.eval_preds = defaultdict(list)

    def plot_eval_preds_and_reset(self, epoch, video_dir):
        """Plots distribution of evaluation predictions.

        Don't forget to clear the `self.eval_preds` at the end! It's meant to be a
        defaultdict to save items, with one item saved per minibatch. Saves the
        original observations as well as the predictios, so we can verify later.
        """
        if not self.args.save_eval_outputs:
           return

        eval_preds_np = np.concatenate(self.eval_preds['act_pol'])
        scaling = (self.args.scale_targets or self.args.scale_pcl_flow)
        suffix = f'fig_net_preds_{str(epoch).zfill(4)}_scaling_{scaling}.png'
        figname = os.path.join(video_dir, suffix)
        utils.plot_action_net_preds(eval_preds_np, figname, scaling=scaling)

        # Also save the original defaultdict to a file.
        if self.args.save_eval_outputs:
            fname = os.path.join(video_dir, f'preds_{str(epoch).zfill(4)}_dict.pkl')
            with open(fname, 'wb') as fh:
                pickle.dump(self.eval_preds, fh)

        self.eval_preds = defaultdict(list)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

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
        out = (input - target) ** 2
        weighted = (self.weights * out).mean()
        return weighted


    def get_demo_flow(self, obs, info=None, act_gt=None):
        '''This function is a modification of the get_single_flow, in the sense that we 
        visualize the flow for the entire evaluation demonstration, rather than just the 
        one single action'''
        assert self.act_type in FLOW_ACTS or self.encoder_type in PCL_COMPRESS
        if not isinstance(obs, torch_geometric.data.Data):
            obs = utils.create_pointnet_pyg_data(obs, self.device)
        if info is not None and not isinstance(info, torch.Tensor):
            info = torch.from_numpy(info)
            info = info.to(torch.float32).to(self.device)

        # Forward pass, assigns flow to `actor.trunk.flow_per_pt`
        act = self.actor(obs, info=info)
        #! Might have to convert this into an bp.where function instead
        #! NOTE(sarthak): Not required, you're just looking at the occurance of the mini-batch number 0 and then then pick only those points from the giant list
        #! of observations
        # My understanding of this is that the indices returned here will be only for the first instance in the batch of size
        # batch_size. Here, I would like to visualize the policy.

        # My proposed modifications include: get the flow for the entire batch_sized actions

        demo_flow_data = defaultdict(list)

        demo_flow_data['batch_size'] = obs.batch.cpu().numpy()[-1] + 1
        demo_flow_data['demo_data'].append(({'demo_size': np.sum(obs.batch.cpu().numpy() == 0), 'start_idx' : 0, 'end_idx': np.sum(obs.batch.cpu().numpy() == 0)}))

        for demo in range(1, demo_flow_data['batch_size']):
            _previous_idxs = demo_flow_data['demo_data'][-1]['end_idx']
            demo_flow_data['demo_data'].append(({'demo_size': np.sum(obs.batch.cpu().numpy() == demo), 'start_idx': _previous_idxs, 'end_idx': np.sum(obs.batch.cpu().numpy() == demo) + _previous_idxs}))

        print('demo_flow_data: ', demo_flow_data['batch_size'], demo_flow_data.keys(), 'demo_size: ', demo_flow_data['demo_data'][31]['demo_size'], 'demo start: ', demo_flow_data['demo_data'][31]['start_idx'], 'demo_end: ', demo_flow_data['demo_data'][31]['end_idx'])

        for demo in range(demo_flow_data['batch_size']):
            start_idx = demo_flow_data['demo_data'][demo]['start_idx']
            end_idx = demo_flow_data['demo_data'][demo]['end_idx']

            # This basically means the points in the observation. For batch == 0, this means 1400 corresponding to that time-step, out of 32
            # pts_in_pcl = np.sum(obs.batch.cpu().numpy() == 0)

            # This is the flow for ALL points in the scene
            flow_1 = self.actor.trunk.flow_per_pt[start_idx:end_idx]
            #! obs.x is just the objectness noted on the pointcloud. Again, one hot-encoded

            # These are the encodings of ALL points in the scene
            tool_1 = obs.x[start_idx:end_idx]
            #! Getting the position of all the points in the same pointcloud

            # These are the XYZ positions of ALL points in the scene
            pos_1 = obs.pos[start_idx:end_idx]

            # These are the indices of JUST the tool in the scene, sorted by the encoding of the scene
            tool_one = torch.where(tool_1[:, 0] == 1)[0]

            # These are the indices of JUST the target points, sorted by the encoding of the scene
            targ_one = torch.where(tool_1[:, 1] == 1)[0]

            # These are the XYZ positions of JUST the tool in the scene
            xyz = pos_1[tool_one]

            demo_flow_data['demo_data'][demo]['xyz'] = xyz

            # These are the XYZ positions of JUST the target in the scene
            xyz_t = pos_1[targ_one]

            # These is the flow of JUST the tool in the scene. Might want to include the target in the future?
            flow = flow_1[tool_one]

            demo_flow_data['demo_data'][demo]['flow'] = flow

            demo_flow_data['demo_data'][demo]['xyz_t'] = xyz_t

            # Adding xyz_t here which is the XYZ location of the target points

            if act_gt is not None:
                # First two cases apply if using PN++ avg where each mb item has 3-dim.
                if act_gt.shape == torch.Size([self.args.batch_size, 3]):
                    act_gt = act_gt[0]
                    return {'xyz': xyz, 'flow': flow, 'act_gt': act_gt.expand(xyz.shape[0], 3), 'xyz_t': xyz_t}
                elif act_gt.shape == torch.Size([3]):
                    return {'xyz': xyz, 'flow': flow, 'act_gt': act_gt.expand(xyz.shape[0], 3), 'xyz_t': xyz_t}
                else:
                    # Grabbing the ground truth flow for just ALL points here
                    # act_gt is (N,flow_dim), just get those from the first one.
                    act_gt_temp = act_gt[start_idx:end_idx]

                # Grabbing the ground truth flow of just the tool points here
                demo_flow_data['demo_data'][demo]['act_gt'] = act_gt_temp[tool_one]
                # return xyz, flow, act_gt[tool_one]

            # Extract actual reported flow if possible
            #! What Daniel/Eddie mean by actual reported flow is just the tool flow, as opposed to flow_per_pt which is the flow for all the points in the scene
            if self.act_type in FLOW_ACTS:
                # This is action that the robot is supposed to take.
                demo_flow_data['demo_data'][demo]['act'] = act[tool_one]

        # return xyz, flow
        return demo_flow_data

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
        #! Might have to convert this into an bp.where function instead
        #! NOTE(sarthak): Not required, you're just looking at the occurance of the mini-batch number 0 and then then pick only those points from the giant list
        #! of observations
        pts_in_pcl = np.sum(obs.batch.cpu().numpy() == 0)
        flow_1 = self.actor.trunk.flow_per_pt[:pts_in_pcl]
        #! obs.x is just the objectness noted on the pointcloud. Again, one hot-encoded
        tool_1 = obs.x[:pts_in_pcl]
        #! Getting the position of all the points in the same pointcloud
        pos_1 = obs.pos[:pts_in_pcl]

        #! Change this to reflect the one hot encoding of the tool
        tool_one = torch.where(tool_1[:, 0] == 1)[0]
        # Isolating target points here
        targ_one = torch.where(tool_1[:, 1] == 1)[0]
        xyz = pos_1[tool_one]
        # We get the actual positions of these indices here
        xyz_t = pos_1[targ_one]
        flow = flow_1[tool_one]

        # Adding xyz_t here which is the XYZ location of the target points
        ret_dict = {
            'xyz': xyz,
            'flow': flow,
            'xyz_t': xyz_t
        }

        if act_gt is not None:
            # First two cases apply if using PN++ avg where each mb item has 3-dim.
            if act_gt.shape == torch.Size([self.args.batch_size, 3]):
                act_gt = act_gt[0]
                return {'xyz': xyz, 'flow': flow, 'act_gt': act_gt.expand(xyz.shape[0], 3), 'xyz_t': xyz_t}
            elif act_gt.shape == torch.Size([3]):
                return {'xyz': xyz, 'flow': flow, 'act_gt': act_gt.expand(xyz.shape[0], 3), 'xyz_t': xyz_t}
            else:
                # act_gt is (N,flow_dim), just get those from the first one.
                act_gt = act_gt[:pts_in_pcl]

            ret_dict['act_gt'] = act_gt[tool_one]
            # return xyz, flow, act_gt[tool_one]

        # Extract actual reported flow if possible
        #! What Daniel/Eddie mean by actual reported flow is just the tool flow, as opposed to flow_per_pt which is the flow for all the points in the scene
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
            if self.encoder_type in PCL_MODELS:
                obs_pn2 = utils.create_pointnet_pyg_data(obs, self.device)
                if info is not None and not isinstance(info, torch.Tensor):
                    info = torch.from_numpy(info)
                    info = info.to(torch.float32).to(self.device)
                act = self.actor(obs_pn2, info=info)
                # NOTE (eddie): We run _flow_to_env_action for all segmentation outputs even if it's a PCL_COMPRESS model. This is because pointnet_svd_centered outputs actions in the wrong frame
                # However, sometimes it might be desirable to take the PCL_COMPRESS outputs. If we'd like to do this, we should implement some flag for it.
                if (self.act_type in FLOW_ACTS or self.encoder_type in PCL_COMPRESS):
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

    def update(self, replay_buffer):
        """Just updates the actor (policy). There's no critic.

        Using MSE since we have continuous actions. The `act_pol` and `act_gt`
        have shapes (batch_size, act_dim) each. Take component-wise difference,
        find L2 diff across 'rows', then average. Only consider tool points!

        Apply a tool mask in the special case if we are using tool flow as the
        action, since the segmented point cloud has indices (rows in np.array)
        corresponding to non-tool points, and we don't do MSE on those. For
        example, this is used in pointwise losses with comparing flow vectors.

        Separate MSEs for pos and rot are for debugging, not actual training.
        """
        obs, act_gt, info = replay_buffer.sample_obs_act(get_info=True)

        # Get output, mask out tool (in both prediction & target) if needed.
        act_pol = self.actor(obs, info=info)
        if self.act_type in FLOW_ACTS:
            act_pol, act_gt = self._mask_out_tools(obs, act_pol, act_gt)

        # Compute (usually) MSE loss.
        if self.use_weighted_R_t_loss:
            actor_loss = self.weighted_R_t_loss(act_pol, act_gt)
        else:
            actor_loss = self.MSE_loss(act_pol, act_gt)

        # Compute consistency loss
        if self.use_consistency_loss:
            assert self.encoder_type in POINTWISE_LOSSES
            if self.actor.trunk.means is None:
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
        return (actor_loss.item(), pos_loss.item(), rot_loss.item(), consistency_loss.item())

    def evaluate_mse(self, obs, act_gt, info=None):
        """For validation, serves as another way to measure BC.

        For PointNet++, again we assume that all have same # of points.
        As with training, only consider the tool points! The separate MSEs
        at the end for pos and rot are for debugging, not actual training.

        Record predictions to plot later. So far, mainly tested for regressing
        directly to 3D or 6D poses, not for flow.
        """
        with torch.no_grad():
            act_pol = self.actor(obs, info=info)
            if self.act_type in FLOW_ACTS:
                act_pol, act_gt = self._mask_out_tools(obs, act_pol, act_gt)

            if self.use_weighted_R_t_loss:
                valid_loss = self.weighted_R_t_loss(act_pol, act_gt)
            else:
                valid_loss = self.MSE_loss(act_pol, act_gt)

            # Get MSEs for each separate component, ignoring weights.
            pos_loss = self.MSE_loss(act_pol[:,:3], act_gt[:,:3])
            if act_pol.shape[1] > 3:
                rot_loss = self.MSE_loss(act_pol[:,3:], act_gt[:,3:])
            else:
                rot_loss = torch.zeros(1)

            # Accumulate evaluation predictions.
            if self.args.save_eval_outputs:
                self.eval_preds['obs_x'].append(obs.x.detach().cpu().numpy())
                self.eval_preds['obs_pos'].append(obs.pos.detach().cpu().numpy())
                self.eval_preds['act_pol'].append(act_pol.detach().cpu().numpy())

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
        assert self.encoder_type not in PCL_COMPRESS, self.encoder_type
        tool_mask = torch.zeros_like(act_gt).to(self.device)
        tool_pts = torch.where(obs.x[:, 0] == 1)[0]  # 0 = tool column
        tool_mask[tool_pts] = 1.  # will create rows of 1s
        act_pol = act_pol * tool_mask
        act_gt = act_gt * tool_mask
        return (act_pol, act_gt)

    def _flow_to_env_action(self, obs, flow, flow_r=None, info=None):
        #! This was the function that I was looking for all this while! I wanted to know how the actual flow_per_pt_r gets
        #! converted to a 6DoF action
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
            obs_tool_idxs = torch.where(obs.x[:, 0] == 1)[0]
            flow_tool_pts = flow[obs_tool_idxs]
            act = torch.mean(flow_tool_pts, dim=0)
            return act

        elif self.args.method_flow2act == 'svd':
            if len(info.shape) > 1:
                info = info.squeeze(0)
            tip_pos_one = info[:3]
            if flow_r is None:
                # 3D flow case
                tool_one = torch.where(obs.x[:, 0] == 1)[0]
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

                xyz_mean = xyz.unsqueeze(0).mean(dim=1, keepdims=True)
                relative_xyz_mean = xyz_mean - tip_pos_one[None, None, :]
                t_correction = relative_xyz_mean - torch.bmm(relative_xyz_mean, rot_matrices)
                trans += t_correction.squeeze(1)

                #pred_eepose = torch.cat([trans, quats], dim=1)
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

                # Convert rotation into axis-angle
                quats = matrix_to_quaternion(matrix=rot_matrices.transpose(1, 2))
                axis_ang = quaternion_to_axis_angle(quaternions=quats)

                # Change rotation frame to be around tip
                xyz_mean = xyz.unsqueeze(0).mean(dim=1, keepdims=True)
                relative_xyz_mean = xyz_mean - tip_pos_one[None, None, :]
                t_correction = relative_xyz_mean - torch.bmm(relative_xyz_mean, rot_matrices)
                mean_one += t_correction.squeeze(1)

                pred_eepose = torch.cat([mean_one, axis_ang], dim=1)
                return pred_eepose.squeeze(0)
        else:
            raise ValueError(self.args.method_flow2act)
