"""Support two types of replay buffers

A 'traditional' buffer where point clouds (PCLs) have same # of points.
Another 'PC' buffer where we can have different # of points in each PCL.
"""
import os
from os.path import join
from operator import itemgetter
import numpy as np
import pickle
import torch
from torch_geometric.data import Data, Batch
from pyquaternion import Quaternion
from pytorch3d.transforms import axis_angle_to_quaternion
from scipy.spatial.transform import Rotation as Rot
from bc.env_specific_hacks import (
    get_episode_part_to_use, env_scaling_targets, get_info_when_sampling,
    get_fake_point_dense_tf, get_rgbd_depth_segm_masks, get_state_info_for_mlp
)
np.set_printoptions(edgeitems=20)
DEG_TO_RAD = np.pi / 180.

from bc.bc import FLOW_ACTS, ALL_ACTS, PCL_MODELS
from bc import utils, rotations
from bc.utils import MixedMediaToolReducer


class BehavioralCloningData():
    """Replay buffer to store data for Behavioral Cloning.

    Two subclasses for this depending on the nature of the point cloud data.
    If using images, keep values between 0 to 255 (encoder later divides by 255).

    Only need to save (obs,act) pairs and not rewards, but we should use a train
    / eval split. In `FlexEnv.reset()`, the last 20% of config indices are set as
    validation. Do something similar here by defining train / eval configs.
    """

    def __init__(self, args, obs_shape, action_shape, info_shape, device,
            action_lb=None, action_ub=None, action_repeat=None, state_as_target=False):
        self.args = args
        self.scale_pcl_flow = args.scale_pcl_flow
        self.scale_pcl_val = args.scale_pcl_val
        self.scale_targets = args.scale_targets
        self.bc_data_dir = args.bc_data_dir
        self.filtered = args.bc_data_filtered
        self.packbits = (args.encoder_type == 'segm')
        self.encoder_type = args.encoder_type
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.capacity = args.data_buffer_capacity
        self.image_size_crop = args.image_size_crop
        self.use_geodesic_dist = args.use_geodesic_dist
        self.data_augm_img = args.data_augm_img
        self.data_augm_PCL = args.data_augm_PCL
        self.dense_transform = args.dense_transform
        self.remove_zeros_PCL = args.remove_zeros_PCL
        self.zero_center_PCL = args.zero_center_PCL
        self.gaussian_noise_PCL = args.gaussian_noise_PCL
        self.reduce_tool_PCL = args.reduce_tool_PCL
        self.reduce_tool_points = args.reduce_tool_points
        self.tool_point_num = args.tool_point_num
        self.obs_shape = obs_shape
        self.info_shape = info_shape
        self.device = device

        # Turn OFF target scaling in this case!
        self.state_as_target = state_as_target
        if self.state_as_target:
            self.scale_targets = False

        assert not (self.scale_pcl_flow and self.scale_targets), 'Do not use both'
        assert not (self.reduce_tool_PCL != 'None' and self.dense_transform), 'Do not use both'
        assert not (self.reduce_tool_points and self.reduce_tool_PCL != 'None'), 'These args are for different environments! Choose one'
        assert not (self.data_augm_PCL != 'None' and self.gaussian_noise_PCL > 0.), 'Choose one'
        assert self.args.env_name in ['PourWater', 'PourWater6D', 'MMOneSphere', 'MMMultiSphere', 'SpheresLadle']

        self.obs_mode = args.env_kwargs['observation_mode']
        self.action_mode = args.env_kwargs['action_mode']  # what SG env used internally
        self.action_type = args.act_type  # the nature of what PN++ should predict
        self.action_lb = action_lb        # per-timestep act component lower bounds
        self.action_ub = action_ub        # per timestep act component upper bounds
        self.action_shape = action_shape
        self.action_repeat = action_repeat

        assert not (self.reduce_tool_points and self.action_type != "ee2flow"), "ee2flow needed for reduce_tool_points"

        # Something new as of 04/25, helps test training sizes.
        self.n_train_demos = args.n_train_demos
        self.n_valid_demos = args.n_valid_demos

        # Handle different action repeats, and how much data we should use per
        # episode; some demonstrators didn't do much near the end of episodes.
        assert self.action_repeat == 8, self.action_repeat
        self.ep_len = 100
        self.max_train_demo = 1000

        # The proprioceptive obs is stored as float32, pixels obs as uint8.
        # If we go the BC from RGBD input, might be easier to use float32.
        if self.packbits:
            raise NotImplementedError()
        elif self.encoder_type in PCL_MODELS:
            assert len(obs_shape) > 1
            self.obs_dtype = np.float32
        elif self.obs_mode in ['cam_rgbd', 'depth_img', 'depth_segm', 'rgb_segm_masks',
                'rgbd_segm_masks', 'state']:
            self.obs_dtype = np.float32
        else:
            self.obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        # Bells and whistles.
        self.idx = 0
        self.last_save = 0
        self.full = False

        # New stuff for BC to handle filtering, since we filter and track the configs
        # which were successful, and should only evaluate on these at test time.
        self.first_idx_train = 0   # (s,a) pair, inclusive
        self.last_idx_train = -1   # (s,a) pair, exclusive
        self.first_idx_valid = -1  # (s,a) pair, inclusive
        self.last_idx_valid = -1   # (s,a) pair, exclusive
        self._train_config_idxs = None  # (filtered) train configs only
        self._valid_config_idxs = None  # (filtered) valid configs only

        # Tool reducer things
        # NOTE(eddieli): A bit of a dirty hack, but we use the [MixedMediaToolReducer]
        # to keep track of the global rotation of our environment. Concretely,
        # we can get the global rotation with [self.tool_reducer.rotation]
        self.tool_reducer = MixedMediaToolReducer(args, self.action_repeat)
        # if self.reduce_tool_points:
        #     self.tool_reducer = MixedMediaToolReducer(args, self.action_repeat)

    @property
    def train_config_idxs(self):
        return self._train_config_idxs

    @property
    def valid_config_idxs(self):
        return self._valid_config_idxs

    def __len__(self):
        return self.capacity

    def num_train_items(self):
        assert self.first_idx_train < self.last_idx_train
        return (self.last_idx_train - self.first_idx_train)

    def get_valid_idx_ranges(self):
        """These refer to replay buffer item idxs, NOT config idxs."""
        assert 0 < self.first_idx_valid < self.last_idx_valid < self.capacity
        return (self.first_idx_valid, self.last_idx_valid)

    def _load_from_data(self):
        """Load BC training data.

        The exact type we load depends on the encoder, as we usually save many
        different types of observations for a fixed demonstrator (to keep things
        consistent). For now we've been saving observations as tuples:
            (keypts, img_rgb, segm_img, pc_array, flow, depth)
        New 08/18/2022: saving depth as the 6th item in the tuples.

        See bc/env_specific_hacks.py for stuff related to scaling targets/flow, etc.
        NOTE: tool flow doesn't take action repeat into account. For scooping, we
        have +/- 0.004 translation bounds as defaults, so act repeat 8 means tool
        flow bounds are +/- 0.032, so divide by 8 here and it should now align with
        ee pose changes, so we can call `env.step(action)` consistently in external code.
        """
        def get_obs_tool_flow(pcl, tool_flow):
            # NOTE(daniel): this is to get (obs,act) encoded correctly + consistently.
            # If `pcl` is segmented point cloud from time t-1, and `tool_flow`
            # is the flow from time t, then their tool points shoud coincide.
            # We will always provide (2000,d)-sized PCLs; in training, can resize.
            pcl_tool = pcl[:,3] == 1
            tf_pts = tool_flow['points']
            tf_flow = tool_flow['flow']
            n_tool_pts_obs = np.sum(pcl_tool)
            n_tool_pts_flow = tf_pts.shape[0]
            # First shapes only equal if: (a) fewer than max pts or (b) no item/distr.
            assert tf_pts.shape[0] <= pcl.shape[0], f'{tf_pts.shape}, {pcl.shape}'
            # assert tf_pts.shape == tf_flow.shape, f'{tf_pts.shape}, {tf_flow.shape}'
            assert n_tool_pts_obs == n_tool_pts_flow, f'{n_tool_pts_obs}, {n_tool_pts_flow}'
            assert np.array_equal(pcl[:n_tool_pts_obs,:3], tf_pts)  # yay :)
            flow_dim = tf_flow.shape[1]
            a = np.zeros((pcl.shape[0], flow_dim))  # all non-tool point rows get 0s
            a[:n_tool_pts_obs] = tf_flow   # actually encode flow for BC purposes
            return (pcl, a)

        # Load pickle paths into list. One item is one demonstration.
        print(f'\nLoading data for Behavioral Cloning: {self.bc_data_dir}')
        pkl_paths = sorted([
            join(self.bc_data_dir,x) for x in os.listdir(self.bc_data_dir)
                if x[-4:] == '.pkl' and 'BC' in x])

        # If filtering, load file which specifies config indices to keep. This is
        # later used in SoftGym since we'll have more configs and need to subsample.
        print(f'Loading {len(pkl_paths)} configs (i.e., episodes) from data.')
        if self.filtered:
            filt_fname = join(self.bc_data_dir, 'BC_data.txt')
            assert os.path.exists(filt_fname), f'{filt_fname}'
            with open(filt_fname, 'rb') as fh:
                config_idxs = [int(l.rstrip()) for l in fh]
            self._filtered_config_idxs = config_idxs
        else:
            self._filtered_config_idxs = [i for i in range(len(pkl_paths))]

        # Handle train and valid _config_ indexes (we only want filtered ones).
        self._train_config_idxs = self._filtered_config_idxs[:self.n_train_demos]
        self._valid_config_idxs = self._filtered_config_idxs[
                self.max_train_demo : self.max_train_demo + self.n_valid_demos]
        print(f'First {self.n_train_demos} idxs of starting configs are training.')
        print(f'Config at filtered index {self.max_train_demo} is first valid episode.')
        # These give the indices for the original (including filtered) set of configs.
        print(f'Train configs (start,end), (inclusive,inclusive): '
            f'{self._train_config_idxs[0]}, {self._train_config_idxs[-1]}')
        print(f'Valid configs (start,end), (inclusive,inclusive): '
            f'{self._valid_config_idxs[0]}, {self._valid_config_idxs[-1]}')

        # Action bounds. Careful about rotations.
        assert self.action_type in ALL_ACTS, self.action_type
        print(f'Action type: {self.action_type}. Act bounds:')
        print(f'  lower: {self.action_lb}')
        print(f'  upper: {self.action_ub}')
        n_diff_ee_flow = 0

        # Iterate through filtered paths, only keeping what we need. We later use
        # indices to restrict the sampling.
        for pidx,pkl_pth in enumerate(pkl_paths):
            if pidx % 50 == 0:
                print(f'  checking episode/config idx: {pidx} on idx {self.idx}')

            # Handle train / valid logic limits.
            if pidx == self.n_train_demos:
                print(f'  finished {pidx} demos, done w/train at idx {self.idx}')
                self.last_idx_train = self.idx
                if pidx < self.max_train_demo:
                    continue
            elif self.n_train_demos < pidx < self.max_train_demo:
                continue
            if pidx == self.max_train_demo:
                print(f'  now on {pidx}, start of valid demos')
                self.first_idx_valid = self.idx
            if pidx == self.max_train_demo + self.n_valid_demos:
                print(f'  on {pidx}, exit now after {self.n_valid_demos} valid demos')
                break

            # Each 'data' is one episode, with 'obs' and 'act' keys.
            with open(pkl_pth, 'rb') as fh:
                data = pickle.load(fh)
            act_key = 'act_raw'
            len_o = len(data['obs'])
            len_a = len(data[act_key])  # TODO(daniel) use act_scaled?
            if len_a == 0:
                # We changed keys from act -> {act_raw,act_scaled} 04/26
                len_a = len(data['act'])
                act_key = 'act'
                print(f'FYI, using outdated key for actions: {act_key}')
            assert len_o == len_a, f'{len_o} vs {len_a}'

            # Now decide on what part of the demo to use. See method for details.
            ep_part_to_use = get_episode_part_to_use(
                env_name=self.args.env_name,
                bc_data_dir=self.args.bc_data_dir,
                ep_len=self.ep_len,
                len_o=len_o)

            # Reset tool reducer
            # if self.reduce_tool_points:
            self.tool_reducer.reset()

            # Add each (obs,act) from this episode into the data buffer.
            # The `obs` is actually a tuple, so extract appropriate item.
            for t in range(ep_part_to_use):
                obs_tuple = data['obs'][t]
                act_raw = data[act_key][t]

                # Process observation.
                if self.encoder_type == 'mlp':
                    obs = get_state_info_for_mlp(obs_tuple, self.args.env_name, self.tool_reducer)
                elif self.encoder_type == 'pixel':
                    obs = np.transpose(obs_tuple[1], (2,0,1))
                    if self.obs_mode == 'cam_rgbd':
                        assert len(obs_tuple) == 6
                        depth_img = obs_tuple[5]  # (H,W)
                        depth_img = depth_img[None,...]
                        obs = np.concatenate(
                            (obs.astype(np.float32), depth_img), axis=0
                        )  # (4,H,W)
                    elif self.obs_mode in ['depth_segm', 'depth_img','rgb_segm_masks',
                            'rgbd_segm_masks']:
                        # Override the `obs`, important! No RGB here.
                        obs = get_rgbd_depth_segm_masks(
                            obs_tuple=obs_tuple,
                            obs_mode=self.obs_mode,
                            obs_rgb=obs,
                            env_name=self.args.env_name,
                        )
                elif self.encoder_type == 'segm':
                    obs = np.transpose(obs_tuple[2], (2,0,1))
                elif self.encoder_type in PCL_MODELS:
                    obs = obs_tuple[3]
                    if self.reduce_tool_PCL != 'None':
                        obs = self.change_obs_reduced_tool(obs, info=obs_tuple[0])
                    if self.reduce_tool_points:
                        obs = self.tool_reducer.reduce_tool(obs, info=obs_tuple[0])
                    # Save 6DoF action for 6DoF rotation models
                    qt_current = self.tool_reducer.rotation
                    self.tool_reducer.step(act_raw)
                else:
                    raise NotImplementedError(self.encoder_type)

                # Process action.
                if self.state_as_target:
                    # Handle this first, in case we use state as targets.
                    # The `act` name is thus highly misleading.
                    if self.args.env_name == 'MMOneSphere':
                        act = obs_tuple[6]  # should have 7 items as of 08/27 for SoftGym
                    elif self.args.env_name in ['PourWater', 'PourWater6D']:
                        act = get_state_info_for_mlp(obs_tuple, self.args.env_name, self.tool_reducer)
                elif self.action_type == 'ee':
                    # If action repeat > 1, this ee is repeated that many times.
                    assert self.action_mode == 'translation', self.action_mode
                    act = act_raw
                elif self.action_type == 'flow':
                    assert (self.action_mode in ['translation',
                        'translation_axis_angle']), self.action_mode
                    # Want the _next_ observation at time t+1. Careful, for action
                    # repeat, flow only considers before/after ALL the repetitions.
                    obs_tuple_next = data['obs'][t+1]
                    tool_flow = obs_tuple_next[4]
                    obs, act = get_obs_tool_flow(obs, tool_flow)
                    # Careful, if using rotations, cannot just divide by act repeat.
                    # With translations, need to divide flow to make it like ee since
                    # we only see the stuff before / after ALL act repetitions.
                    if self.action_mode == 'translation':
                        act = act / self.action_repeat
                    elif self.action_mode == 'translation_axis_angle':
                        assert self.action_repeat == 1
                elif self.action_type == 'ee2flow':
                    # Used for "intended" non-action repeat flow
                    if act_raw.shape[0] == 3:
                        # NOTE(daniel) 06/21: I think this code is wrong, we should not
                        # divide by action_ub here as we do that later when we call add()!
                        # But for translation-only data let's just use 'flow' as act type.
                        # translation
                        # Debugging: if there's any discrepancy we will use what `act_raw`
                        # gives us and format it as flow data. If this works, then the
                        # segmentation PN++ is fine and we need to fix flow data somehow?
                        obs_tuple_next = data['obs'][t+1]
                        tool_flow = obs_tuple_next[4]
                        obs, act = get_obs_tool_flow(obs, tool_flow)
                        act = act * (1.0 / (self.action_ub * self.action_repeat))
                        tool_idxs = np.where(obs[:,3] == 1)[0]
                        act_flow_avg = np.mean(act[tool_idxs], axis=0)
                        act_ee_delta = (act_raw) * (1.0 / self.action_ub)
                        if not np.allclose(act_flow_avg, act_ee_delta):
                            # The flow should have smaller abs value due to clipping.
                            for k in range(len(act_flow_avg)):
                                assert np.abs(act_flow_avg[k]) <= np.abs(act_ee_delta[k])+1e-7, \
                                    f'k={k}, {act_flow_avg} vs {act_ee_delta}'
                            act[tool_idxs] = act_ee_delta
                            n_diff_ee_flow += 1
                    elif act_raw.shape[0] == 6:
                        assert self.action_mode == 'translation_axis_angle', self.action_mode

                        # Get global delta action
                        act_tran, tool_origin, delta_quat = rotations.convert_action(
                            "flow",
                            self.args.env_name,
                            obs_tuple,
                            act_raw,
                            qt_current
                        )

                        # New case, if we reduce it, then we don't bother with querying
                        # the obs_tuple_next as that was only to get a set of 'prev' tool
                        # points (for the _current_ obs!) to generate flow.
                        if self.reduce_tool_PCL == 'pouring_v01':
                            n_pts = 10
                            prev_tool_points = obs[:n_pts, :3]
                        elif self.reduce_tool_points:
                            n_pts = self.tool_point_num
                            prev_tool_points = obs[:n_pts, :3]
                        else:
                            # Once again this info is not necessary.
                            obs_tuple_next = data['obs'][t+1]
                            tool_flow = obs_tuple_next[4]

                            # We use `prev_tool_points` since that contains the tool points, but
                            # this can be equivalently extracted from current obs (see assertion).
                            # We do NOT use `tool_flow['flow']` here due to action repeat issues.
                            prev_tool_points = tool_flow['points']
                            n_pts = prev_tool_points.shape[0]
                            assert np.array_equal(prev_tool_points, obs[:n_pts, :3])

                        # Back to creating flow info.
                        ee2flow = np.zeros_like(prev_tool_points)
                        ee2flow += act_tran

                        delta_quat._normalise()
                        dqp = delta_quat.conjugate.q

                        relative = prev_tool_points - tool_origin

                        vec_mat = np.zeros((n_pts, 4, 4), dtype=prev_tool_points.dtype)
                        vec_mat[:, 0, 1] = -relative[:, 0]
                        vec_mat[:, 0, 2] = -relative[:, 1]
                        vec_mat[:, 0, 3] = -relative[:, 2]

                        vec_mat[:, 1, 0] = relative[:, 0]
                        vec_mat[:, 1, 2] = -relative[:, 2]
                        vec_mat[:, 1, 3] = relative[:, 1]

                        vec_mat[:, 2, 0] = relative[:, 1]
                        vec_mat[:, 2, 1] = relative[:, 2]
                        vec_mat[:, 2, 3] = -relative[:, 0]

                        vec_mat[:, 3, 0] = relative[:, 2]
                        vec_mat[:, 3, 1] = -relative[:, 1]
                        vec_mat[:, 3, 2] = relative[:, 0]

                        mid = np.matmul(vec_mat, dqp)
                        mid = np.expand_dims(mid, axis=-1)

                        relative_rot = delta_quat._q_matrix() @ mid
                        relative_rot = relative_rot[:, 1:, 0]

                        ee2flow += relative_rot - relative

                        obs, act = get_obs_tool_flow(obs, {
                            'points': prev_tool_points,
                            'flow': ee2flow,
                        })

                        # Debugging flow visuals only to check that ee2flow has correct
                        # meaning. Mainly for PourWater (e.g., if flow points in pouring
                        # direction). Pick which episodes to plot with `pidx` etc.
                        #if dtheta > 0:
                        #if pidx in [0,1,2,3]:
                        #    utils.create_pw_flow_plot_debug(
                        #            pts=prev_tool_points,
                        #            flow=ee2flow,
                        #            args=self.args,
                        #            hang=False,
                        #            enforce_scaling=False,
                        #            time_step=t,
                        #            pidx=pidx,
                        #    )
                    else:
                        raise NotImplementedError(f'Unrecognized action shape {act_raw.shape}')
                elif self.action_type == 'ee2flow_sep_rt':
                    assert act_raw.shape[0] == 6, "Expected 6-dim action"

                    # Get global delta action
                    act_tran, tool_origin, delta_quat = rotations.convert_action(
                        "flow",
                        self.args.env_name,
                        obs_tuple,
                        act_raw,
                        qt_current
                    )

                    # New case, if we reduce it, then we don't bother with querying
                    # the obs_tuple_next as that was only to get a set of 'prev' tool
                    # points (for the _current_ obs!) to generate flow.
                    if self.reduce_tool_PCL == 'pouring_v01':
                        n_pts = 10
                        prev_tool_points = obs[:n_pts, :3]
                    elif self.reduce_tool_points:
                        n_pts = self.tool_point_num
                        prev_tool_points = obs[:n_pts, :3]
                    else:
                        # Once again this info is not necessary.
                        obs_tuple_next = data['obs'][t+1]
                        tool_flow = obs_tuple_next[4]

                        # We use `prev_tool_points` since that contains the tool points, but
                        # this can be equivalently extracted from current obs (see assertion).
                        # We do NOT use `tool_flow['flow']` here due to action repeat issues.
                        prev_tool_points = tool_flow['points']
                        n_pts = prev_tool_points.shape[0]
                        assert np.array_equal(prev_tool_points, obs[:n_pts, :3])

                    ee2flow_t = np.zeros_like(prev_tool_points)
                    ee2flow_t += act_tran

                    ee2flow_r = np.zeros_like(prev_tool_points)
                    delta_quat._normalise()
                    dqp = delta_quat.conjugate.q

                    relative = prev_tool_points - tool_origin

                    vec_mat = np.zeros((n_pts, 4, 4), dtype=prev_tool_points.dtype)
                    vec_mat[:, 0, 1] = -relative[:, 0]
                    vec_mat[:, 0, 2] = -relative[:, 1]
                    vec_mat[:, 0, 3] = -relative[:, 2]

                    vec_mat[:, 1, 0] = relative[:, 0]
                    vec_mat[:, 1, 2] = -relative[:, 2]
                    vec_mat[:, 1, 3] = relative[:, 1]

                    vec_mat[:, 2, 0] = relative[:, 1]
                    vec_mat[:, 2, 1] = relative[:, 2]
                    vec_mat[:, 2, 3] = -relative[:, 0]

                    vec_mat[:, 3, 0] = relative[:, 2]
                    vec_mat[:, 3, 1] = -relative[:, 1]
                    vec_mat[:, 3, 2] = relative[:, 0]

                    mid = np.matmul(vec_mat, dqp)
                    mid = np.expand_dims(mid, axis=-1)

                    relative_rot = delta_quat._q_matrix() @ mid
                    relative_rot = relative_rot[:, 1:, 0]

                    ee2flow_r += relative_rot - relative

                    ee2flow = np.concatenate([ee2flow_t, ee2flow_r], axis=1)

                    obs, act = get_obs_tool_flow(obs, {
                        'points': prev_tool_points,
                        'flow': ee2flow,
                    })
                elif self.action_type == 'eepose':
                    assert self.action_mode in ['translation_axis_angle'], self.action_mode
                    act = act_raw
                elif self.action_type == 'eepose_convert':
                    assert self.action_mode in ['translation_axis_angle'], self.action_mode
                    # Convert rotation
                    act_tran, tool_origin, converted_rotation = rotations.convert_action(
                        self.args.rotation_representation,
                        self.args.env_name,
                        obs_tuple,
                        act_raw,
                        qt_current
                    )
                    act = np.concatenate((act_tran, converted_rotation))
                else:
                    raise NotImplementedError(self.action_type)

                # Pull keypoints from observation
                keypoints = obs_tuple[0]

                # Finally, add values. Any scaling is done here.
                self.add(obs=obs, action=act, info=keypoints)

        self.last_idx_valid = self.idx

        # Debugging. Also keep self.capacity above self.idx please!
        assert not self.full and self.idx < self.capacity
        print(f'\nDone loading BC data. Buffer full? {self.full}')
        print(f'  idx: {self.idx} out of cap: {self.capacity}')
        print(f'  scaling? {self.scale_pcl_flow}, using {self.scale_pcl_val}')
        print(f'  scaling targets? {self.scale_targets} only transl. (unless MMOneSphere_v02)')
        print(f'  Train idxs for (s,a) pairs (incl, excl): '
            f'({self.first_idx_train},{self.last_idx_train})')
        print(f'  Valid idxs for (s,a) pairs (incl, excl): '
            f'({self.first_idx_valid},{self.last_idx_valid})')
        print(f'  action_repeat: {self.action_repeat}')
        self._debug_print()
        print()


class ImageReplayBuffer(BehavioralCloningData):
    """Replay buffer to store data of equal dimensions.

    Even though we could use equal sized point clouds, keep all point cloud
    data in the `PointCloudReplayBuffer`. Assume this is for images only.
    Store the 128x128 images, but sample 100x100 for training (and at test
    time, only do center-crop sampling).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.obses = np.empty((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.info = np.empty((self.capacity, *self.info_shape), dtype=np.float32)
        self._load_from_data()

    def add(self, obs, action, info):
        """Add the observation and action (and possibly info) pair.

        Not dealing with point cloud data here, so don't check if we need to scale.
        Scaling here is mainly for the naive {RGB -> CNN -> EE pose} method.

        Earlier we were only explicitly scaling translation. However, I think the
        rotations should also be scaled, we should just do action / self.action_ub.
        This should not affect any mixed media results since the action_ub was 1
        for all the rotation parts, but for PourWater it's 0.015 for each.
        """
        if self.scale_pcl_flow:
            raise NotImplementedError()
        elif self.scale_targets:
            action = env_scaling_targets(
                act=action,
                act_ub=self.action_ub,
                env_name=self.args.env_name,
                env_version=self.args.env_version,
                load=True
            )

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.info[self.idx], info)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_obs_act(self, train=True, v_min_i=-1, v_max_i=-1, get_info=False):
        """Standard sampling, but for BC, we support train / valid split.

        I don't think we will ever need `get_info` but if we do be careful about
        which env we are using, as `info` medaning and shape might vary, etc.
        """
        if train:
            min_i = self.first_idx_train
            max_i = self.last_idx_train
            idxs = np.random.randint(min_i, max_i, size=self.batch_size)
        else:
            assert 0 < v_min_i < v_max_i <= self.idx
            assert self.first_idx_valid <= v_min_i < self.last_idx_valid, v_min_i
            assert self.first_idx_valid < v_max_i <= self.last_idx_valid, v_max_i
            idxs = np.arange(v_min_i, v_max_i)

        # Center crop: Bx3x128x128 --> Bx3x100x100, possibly w/data augmentation.
        if self.data_augm_img == 'random_crop':
            obses = utils.random_crop(self.obses[idxs], self.image_size_crop)
        else:
            obses = utils.center_crop_image_mb(self.obses[idxs], self.image_size_crop)
        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        if get_info:
            info = torch.as_tensor(self.info[idxs], device=self.device)
            return obses, actions, info

        return obses, actions

    def _debug_print(self):
        """Might help understand the BC data distribution a bit better."""
        acts = self.actions[:self.idx]
        print(f'  obs: {self.obses.shape}')
        print(f'  act: {self.actions.shape}')
        print(f'  act: {acts.shape} (just the ones we use if using all loaded demos)')
        print(f'  action statistics for the _scaled_ variant (i.e., BC targets):')
        if self.action_type in FLOW_ACTS:
            print(f'  obs mean (pcl) axis 0: {np.mean(self.obses[:,:,0]):0.4f}')
            print(f'  obs mean (pcl) axis 1: {np.mean(self.obses[:,:,1]):0.4f}')
            print(f'  obs mean (pcl) axis 2: {np.mean(self.obses[:,:,2]):0.4f}')
            acts = np.reshape(acts, (acts.shape[0]*acts.shape[1],3))
            print(f'    if concatenating all flow actions: {acts.shape}')
        print(f'    min, max:  {np.min(acts,axis=0)}, {np.max(acts,axis=0)}')
        print(f'    mean,medi: {np.mean(acts,axis=0)}, {np.median(acts,axis=0)}')
        print(f'    std:       {np.std(acts,axis=0)}')


class StateReplayBuffer(ImageReplayBuffer):
    """Now let's use this for state-based info.

    Subclass ImageReplayBuffer, override just the adding method.
    Sorry the subclass naming is bad but it is basically the image replay
    buffer except it doesn't do image data augmentation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample_obs_act(self, train=True, v_min_i=-1, v_max_i=-1, get_info=False):
        """Same as ImageReplayBuffer but no image data augmentation."""
        if train:
            min_i = self.first_idx_train
            max_i = self.last_idx_train
            idxs = np.random.randint(min_i, max_i, size=self.batch_size)
        else:
            assert 0 < v_min_i < v_max_i <= self.idx
            assert self.first_idx_valid <= v_min_i < self.last_idx_valid, v_min_i
            assert self.first_idx_valid < v_max_i <= self.last_idx_valid, v_max_i
            idxs = np.arange(v_min_i, v_max_i)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        if get_info:
            info = torch.as_tensor(self.info[idxs], device=self.device)
            return obses, actions, info

        return obses, actions


class PointCloudReplayBuffer(BehavioralCloningData):
    """Replay buffer to store point cloud data (s,a) pairs.

    Big difference from the normal replay buffer: we allow for a variable
    number of points per PCL, assuming `self.remove_zeros_PCL = True`. (It
    probably should be True.)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # We can have PCL but not necessarily have actions in 2D arrays.
        self.use_PCL_act_list = self.action_type in ['flow', 'ee2flow', 'ee2flow_sep_rt']

        # Support a variable amount of points in point clouds (also for actions!).
        self.obses = []
        if self.use_PCL_act_list:
            self.actions = []
        else:
            self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.info = np.empty((self.capacity, *self.info_shape), dtype=np.float32)

        # Load the data, potentially plot distribution of actions up to `self.idx`.
        self._load_from_data()

        if isinstance(self.actions, np.ndarray):
            acts = self.actions[:self.idx]
            plot_action_hist_buffer(acts)

    def add(self, obs, action, info):
        """Add the observation and action (and possibly info) pair.

        New stuff here compared to other replay buffer: (1) removing 0s from PCL,
        and possibly from actions (only if actions are dense and aligned with the
        PCL), (2) storing as Data types so we can make minibatches later.

        To clarify, we might still later subsample from these point clouds due
        to only extracting tool flow pts (and ignoring items & distractors), and
        this here is just for getting rid of 0s (which always seems good to do).

        If actions represent flow, then there will still be rows of 0s if those
        rows correspond to non-tool objects.

        Earlier we were only explicitly scaling translation:
            action[:3] = action[:3] / self.action_ub[:3]
        Which we used for 3D and 6D action vectors (there's no other option for
        actions UNLESS we're dealing with flow-based actions).

        However, I think we should just do action / self.action_ub. This should
        not affect earlier mixed media results since the action_ub was 1 for all
        the rotation parts, but for PourWater it's 0.015 for each. For flow I
        think we only use `scale_pcl_flow` and not `scale_targets`.

        Also supports (1) noisy data, (2) data augmentation. For (1) inject Gauss
        noise once to each data point and fixed ahead of time. Do this BEFORE any
        scaling. For (2) we'd apply noise or some augmentation EACH time the data
        is sampled (not here). For simplicity, let's not do (1) and (2) together.

        07/17/2022: supports zero-centering the point cloud. We might do this at
        the last stage of processing? Note that this would be before any data augm
        as that happens when sampling data.
        """

        # Since the env has given zero-padded point clouds, remove zeros.
        if self.remove_zeros_PCL:
            tool_idxs = np.where(obs[:,3] == 1)[0]
            targ_idxs = np.where(obs[:,4] == 1)[0]
            if obs.shape[1] == 6:
                dist_idxs = np.where(obs[:,5] == 1)[0]
            else:
                dist_idxs = np.array([])
            n_nonzero_pts = len(tool_idxs) + len(targ_idxs) + len(dist_idxs)

            # Clear out 0s in observation (if any) and in actions (if applicable).
            if n_nonzero_pts < obs.shape[0]:
                nonzero_idxs = np.concatenate(
                        (tool_idxs, targ_idxs, dist_idxs)).astype(np.uint64)
                obs = obs[nonzero_idxs]
                if self.use_PCL_act_list:
                    action = action[nonzero_idxs]

        # First 3 columns are xyz positions in meters, usually quite small.
        if self.gaussian_noise_PCL > 0:
            obs[:, :3] += np.random.normal(0,
                    self.gaussian_noise_PCL, size=obs[:,:3].shape)

        if self.scale_pcl_flow:
            obs[:, :3] *= self.scale_pcl_val
            action *= self.scale_pcl_val  # applies for EE translations and flow
            if self.args.env_name in ['PourWater', 'PourWater6D']:
                # Shape (10,14), with first 6 columns encoding position info.
                info[:, :6] *= self.scale_pcl_val
            elif self.args.env_name in ['MMOneSphere', 'MMMultiSphere', 'SpheresLadle']:
                # Shape (8,), with first 7 parts encoding position info.
                info[:7] *= self.scale_pcl_val
        elif self.scale_targets:
            # Scale transl. parts to (-1,1), assumes symmetrical bounds!
            # Either action is (3,) or (6,), or (N,3) for flow. Actually I'm
            # not sure this is ever used anymore for flow.
            if len(action.shape) == 1:
                action = env_scaling_targets(
                    act=action,
                    act_ub=self.action_ub,
                    env_name=self.args.env_name,
                    env_version=self.args.env_version,
                    load=True
                )
            else:
                assert len(action.shape) == 2 and action.shape[1] == 3, action
                raise ValueError('Check if this case should be used!')
                action[:,:3] = action[:,:3] / self.action_ub[:3]

        # Add synthetic point to 1st row of PCL, overriding existing point.
        if self.dense_transform:
            fake_point_np = get_fake_point_dense_tf(
                    obs=obs, info=info, env_name=self.args.env_name)
            obs[0] = fake_point_np

        # Last processing step before PyGeom, zero-center PCL individually.
        if self.zero_center_PCL:
            centroid_mean = np.mean(obs[:,:3], axis=0)
            obs[:,:3] = obs[:,:3] - centroid_mean

        # Store as `Data` types so `Batch.from_data_list()` works later
        obs_dict = {
            'x': torch.from_numpy(obs[:, 3:]).float(),
            'pos': torch.from_numpy(obs[:, :3]).float(),
        }
        obs = Data.from_dict(obs_dict)

        # For now let's not deal with the complexity of excess capacity.
        assert len(self.obses) < self.capacity, 'Should not exceed capacity in BC'
        self.obses.append(obs)
        if self.use_PCL_act_list:
            self.actions.append(action)
        else:
            np.copyto(self.actions[self.idx], action)
        np.copyto(self.info[self.idx], info)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_obs_act(self, train=True, v_min_i=-1, v_max_i=-1, get_info=False):
        """Standard sampling, but for BC, we support train / valid split.

        Sample observation using `Batch.from_data_list()` which automatically
        forms a minibatch with Data(x, pos, batch, ptr) set up for us. We might
        also need to deal with actions of different dimensions but for those we
        can stack them as they don't get directly passed to a PointNet++.

        We do need `get_info` for many of the SVD / pointwise based methods as
        those require knowing the center of rotation. For scooping we get array
        of shape (M,8), slice to get first 3. For pouring, it's shape (M,10,14)
        so just get the first shape, then the first 3 values to get the bottom
        floor's center coordinate.

        07/12/2022: support data augm for sim PCLs (we were using it in real).
        """
        if train:
            min_i = self.first_idx_train
            max_i = self.last_idx_train
            idxs = np.random.randint(min_i, max_i, size=self.batch_size)
        else:
            assert 0 < v_min_i < v_max_i <= self.idx
            assert self.first_idx_valid <= v_min_i < self.last_idx_valid, v_min_i
            assert self.first_idx_valid < v_max_i <= self.last_idx_valid, v_max_i
            idxs = np.arange(v_min_i, v_max_i)

        # Might need to sample from lists (instead of np arrays).
        obses = list(itemgetter(*idxs)(self.obses))
        obses = Batch.from_data_list(obses).to(self.device)

        if self.use_PCL_act_list:
            # Only used if the actions might be different sized (e.g., with flow).
            actions = list(itemgetter(*idxs)(self.actions))
            actions = np.concatenate(actions)
            actions = torch.from_numpy(actions).to(self.device)
            assert actions.shape[0] == obses.pos.shape[0], actions.shape
        else:
            actions = torch.as_tensor(self.actions[idxs], device=self.device)
            if self.use_geodesic_dist:
                # Special case if we use geodesic dist. Our rotations are stored
                # as axis-angle but we actually want the targets as quaternions.
                # Empirically, returns normalized quaternions (good).
                assert actions.shape[1] == 6, actions.shape
                actions = torch.cat(
                    [actions[:,:3], axis_angle_to_quaternion(actions[:,3:])],
                    dim=1
                )
            assert actions.shape[0] == len(obses.ptr)-1, actions.shape

        # Handle data augmentation! Only for training, of course.
        if train:
            self._augment_PCL(obses, actions)

        if get_info:
            info = torch.as_tensor(self.info[idxs], device=self.device)
            info = get_info_when_sampling(env_name=self.args.env_name, info=info)
            return obses, actions, info

        return obses, actions

    def get_first_obs_act_info(self):
        """Just for debugging flow visualizations.

        MB size 1. If using 3DoF w/alg demonstrator v05, then 1st item has
        action (0,0,0) so use the second one, which is (0,-1,0) if scaling.
        (Using idx=1 assumes training come first in replay buffer ordering.)
        Use `Batch.from_data_list()` to allow forward passes to work.
        """
        ii = 1
        obses = Batch.from_data_list([self.obses[ii]]).to(self.device)
        actions = torch.as_tensor(self.actions[ii], device=self.device)
        info = torch.as_tensor(self.info[ii], device=self.device)
        return obses, actions, info

    def _debug_print(self):
        """Might help understand the BC data distribution a bit better.

        Note that `self.obses` is a list of torch_geometric `Data` objects.
        """
        acts = self.actions[:self.idx]

        # Observations
        num_pts = [data.x.shape[0] for data in self.obses]
        print(f'  #obs: {len(self.obses)}, each is a PCL of possibly different size')
        print(f'  obs[0] avgs: {torch.mean(self.obses[0].pos, dim=0)}')
        print(f'    data.x shape: {np.mean(num_pts):0.2f} +/- {np.std(num_pts):0.1f}')
        print(f'    #pts min max: {np.min(num_pts)}, {np.max(num_pts)}')
        num_pts_t = [len(np.where(data.x[:,0] == 1)[0]) for data in self.obses]
        num_pts_b = [len(np.where(data.x[:,1] == 1)[0]) for data in self.obses]
        num_pts_d = [0.]
        if (self.obses[0]).x.shape[1] > 2:
            num_pts_d = [len(np.where(data.x[:,2] == 1)[0]) for data in self.obses]
        print(f'    # tool pts: {np.mean(num_pts_t):0.2f} +/- {np.std(num_pts_t):0.1f}')
        print(f'    # ball pts: {np.mean(num_pts_b):0.2f} +/- {np.std(num_pts_b):0.1f}')
        print(f'    # dist pts: {np.mean(num_pts_d):0.2f} +/- {np.std(num_pts_d):0.1f}')
        print(f'    min max (tool): {np.min(num_pts_t)}, {np.max(num_pts_t)}')
        print(f'    min max (ball): {np.min(num_pts_b)}, {np.max(num_pts_b)}')
        print(f'    min max (dist): {np.min(num_pts_d)}, {np.max(num_pts_d)}')

        # # Compute lengths? This is mainly to get a number for sigma (noise)
        # # in comparision to the length of the boxes in SoftGym.
        # all_tool = []
        # for data in self.obses:
        #     tool_idx = np.where(data.x[:,0] == 1)[0]
        #     tool_pos = (data.pos[tool_idx]).detach().cpu().numpy() / 250.
        #     tool_min = np.min(tool_pos, axis=0)
        #     tool_max = np.max(tool_pos, axis=0)
        #     all_tool.append( tool_max[0]-tool_min[0] )
        # print(np.mean(all_tool))
        # import pdb; pdb.set_trace()

        # Actions
        if isinstance(self.actions, np.ndarray):
            print(f'  act: {self.actions.shape} (really {acts.shape}), type {type(self.actions)}')
            if self.action_type in FLOW_ACTS:
                acts = np.reshape(acts, (acts.shape[0]*acts.shape[1],3))
                print(f'    if concatenating all flow actions: {acts.shape}')
            print(f'    min, max:  {np.min(acts,axis=0)}, {np.max(acts,axis=0)}')
            print(f'    mean,medi: {np.mean(acts,axis=0)}, {np.median(acts,axis=0)}')
            print(f'    std:       {np.std(acts,axis=0)}')
        else:
            print(f'  #act: {len(self.actions)}, type {type(self.actions)}')
            if self.action_type in FLOW_ACTS:
                a_max = np.array([np.max(act, axis=0) for act in self.actions])
                a_min = np.array([np.min(act, axis=0) for act in self.actions])
                print(f'  max flow: {np.max(a_max, axis=0)}')
                print(f'  min flow: {np.min(a_min, axis=0)}')

    def _augment_PCL(self, obses, actions):
        """Doing data augmentation for point clouds, during training.

        Test in sim here (and do something similar for real). Compare performance
        as a function of different data augmentation levels, on the same test set.
        Pass in the `obses` and `actions` that were sampled; modifying these in
        place here should adjust the data correctly.

        Supported data augmentations:

            [not supported] rot_X_gaussian_Y:
                rotation augm with angles (-X,+X), Gaussian augm with std Y.
            transl_X_gaussian_Y:
                translation augm with per-component (-X,+X) bounds, Gaussian augm.
            gaussian_Y:
                Gaussian augm with std Y

        Caution: the Gaussian noise will depend on if we scaled the original scale
        of the point clouds (which we usually do). Look at `self.scale_pcl_val`.
        """
        if 'rot_' in self.data_augm_PCL:
            angle = float(self.data_augm_PCL.split('_')[1])

            # Rotation augmentation. How to vectorize? TODO(daniel): this isn't quite
            # ready for PourWater because we have to adjust the rotation center. The
            # rotation center assumes the tool tip which is for the ladle / scooping.
            # AND I now realize this might not make sense for pouring as the demos
            # show rotation to a fixed orientation; if we rotate that makes labels wrong.

            for i in range(len(obses.ptr)-1):
                idx1 = obses.ptr[i].detach().cpu().numpy().item()
                idx2 = obses.ptr[i+1].detach().cpu().numpy().item()
                obs_raw = torch.clone(obses.pos[idx1:idx2])  # PCL's xyz (tool+item).
                tool_one = torch.where(obses.x[idx1:idx2, 0] == 1)[0]
                tool_raw = obs_raw[tool_one]

                # Do this at the tip, so the maximum y coord (column index 1).
                max_y = torch.where(tool_raw[:,1] == torch.max(tool_raw[:,1]))[0]
                max_y = max_y[0]  # a torch tensor, pick first idx.
                obs_raw_mean = tool_raw[max_y]
                obs_raw_cent = obs_raw - obs_raw_mean

                # Rotate about the z-axis. See our debug visualizations.
                # Actually we probably want the y-axis in sim, not the z-axis.
                angle_deg = np.random.uniform(-angle, angle)
                angle_rad = angle_deg * DEG_TO_RAD
                angles = np.array([0, angle_rad, 0])
                obs_aug, R = rotate_pc(
                        obs=obs_raw_cent,
                        angles=angles,
                        device=self.device,
                        return_rot=True,
                )

                # MODIFY OBS IN PLACE (back to uncentered raw version).
                obs_aug = obs_aug + obs_raw_mean
                obses.pos[idx1:idx2] = obs_aug

                # Now adjust actions.
                if self.use_PCL_act_list:
                    # The `self.actions` is (N,3) where N=number of points, and has flow.
                    flow_raw = torch.zeros_like(obs_aug)
                    flow_raw += actions[idx1:idx2]
                    flow_aug = torch.zeros_like(obs_aug)
                    flow_aug += actions[idx1:idx2] @ R.double()
                    actions[idx1:idx2] = flow_aug
                    sizeref = 1.0  # actual sized flow vectors
                else:
                    # The `self.actions` is (B,3) where B is batch size.
                    # Applies for naive classif. PN++ and PN++ w/averaging.
                    flow_raw = torch.zeros_like(obs_aug)
                    flow_raw += actions[i]
                    flow_aug = torch.zeros_like(obs_aug)
                    flow_aug += actions[i]
                    flow_aug = flow_aug @ R
                    actions[i] = actions[i] @ R
                    # Decrease length of flow vectors if we scaled by 1/0.004. This will
                    # thus reflect the raw size in the data, which _might_ be 'composed'.
                    sizeref = 0.004

                ## Uncomment to visualize and debug flow, point clouds, data augmentation.
                #fig_raw = utils.pcl_data_aug_viz(
                #        args=self.args,
                #        pts_raw=obs_raw.detach().cpu().numpy(),
                #        pts_aug=obs_aug.detach().cpu().numpy(),
                #        flow_raw=flow_raw.detach().cpu().numpy(),
                #        flow_aug=flow_aug.detach().cpu().numpy(),
                #        tool_pts=tool_one.detach().cpu().numpy(),
                #        sizeref=sizeref,
                #)
                #viz_pth = join('tmp', f'flow_ang_{angle_deg:0.1f}.html')
                #fig_raw.write_html(viz_pth)
                #print(f'See visualization: {viz_pth}')
                #import pdb; pdb.set_trace()

            # Additionally apply Gaussian noise if desired. Use the last idx as usual.
            if 'gaussian_' in self.data_augm_PCL:
                var = float(self.data_augm_PCL.split('_')[-1])
                gauss_noise = ((var**0.5) * torch.randn(*obses.pos.shape)).to(self.device)
                obses.pos += gauss_noise

        elif 'transl_' in self.data_augm_PCL:
            lim = float(self.data_augm_PCL.split('_')[1])

            for i in range(len(obses.ptr)-1):
                idx1 = obses.ptr[i].detach().cpu().numpy().item()
                idx2 = obses.ptr[i+1].detach().cpu().numpy().item()
                obs_raw = torch.clone(obses.pos[idx1:idx2])  # PCL's xyz (tool+item).
                tool_one = torch.where(obses.x[idx1:idx2, 0] == 1)[0]

                # Now randomize and modify obs.
                rand_t = np.random.uniform(-lim, lim, size=3)
                rand_t = torch.from_numpy(rand_t).to(self.device)
                obs_aug = obs_raw + rand_t
                obses.pos[idx1:idx2] = obs_aug

                # Adjust actions. Actually if it's translation only, should be the same?
                if self.use_PCL_act_list:
                    # The `self.actions` is (N,3) where N=number of points, and has flow.
                    flow_raw = torch.clone(actions[idx1:idx2])
                    flow_aug = torch.clone(actions[idx1:idx2])
                    #actions[idx1:idx2] = flow_aug  # we don't have to change this?
                    sizeref = 1.0  # actual sized flow vectors
                else:
                    raise NotImplementedError()
                    # The `self.actions` is (B,3) where B is batch size.
                    # Applies for naive classif. PN++ and PN++ w/averaging.
                    flow_raw = torch.zeros_like(obs_aug)
                    flow_raw += actions[i]
                    flow_aug = torch.zeros_like(obs_aug)
                    flow_aug += actions[i]
                    flow_aug = flow_aug @ R
                    actions[i] = actions[i] @ R
                    # Decrease length of flow vectors if we scaled by 1/0.004. This will
                    # thus reflect the raw size in the data, which _might_ be 'composed'.
                    sizeref = 0.004

                ## Uncomment to visualize and debug flow, point clouds, data augmentation.
                #fig_raw = utils.pcl_data_aug_viz(
                #        args=self.args,
                #        pts_raw=obs_raw.detach().cpu().numpy(),
                #        pts_aug=obs_aug.detach().cpu().numpy(),
                #        flow_raw=flow_raw.detach().cpu().numpy(),
                #        flow_aug=flow_aug.detach().cpu().numpy(),
                #        tool_pts=tool_one.detach().cpu().numpy(),
                #        sizeref=sizeref,
                #)
                #viz_pth = join('tmp',
                #    f'flow_rand_{rand_t[0]:0.3f}_{rand_t[1]:0.3f}_{rand_t[2]:0.3f}.html')
                #fig_raw.write_html(viz_pth)
                #print(f'See visualization: {viz_pth}')
                #import pdb; pdb.set_trace()

            # Additionally apply Gaussian noise if desired. Use the last idx as usual.
            if 'gaussian_' in self.data_augm_PCL:
                var = float(self.data_augm_PCL.split('_')[-1])
                gauss_noise = ((var**0.5) * torch.randn(*obses.pos.shape)).to(self.device)
                obses.pos += gauss_noise

        elif 'gaussian_' in self.data_augm_PCL:
            # Small amount of Gaussian noise each time.
            var = float(self.data_augm_PCL.split('_')[-1])
            gauss_noise = ((var**0.5) * torch.randn(*obses.pos.shape)).to(self.device)
            obses.pos += gauss_noise

        else:
            assert self.data_augm_PCL == 'None', self.data_augm_PCL

    def change_obs_reduced_tool(self, obs, info):
        """Changes observation so we switch to fewer tool points.

        Mainly to test if using fewer tool points will help.

        The `info` comes from the observation tuple and we assume it has enough
        keypoint-related information. It is env-dependent.
        """
        if self.reduce_tool_PCL == 'pouring_v01':
            # `info` is the (10,14)-shaped pyflex shapes. The first 5 rows are the
            # tool, the next 5 are the target glass. At test time we need to call the
            # env to query this. Fortunately we saved it during BC data collection.
            assert info.shape == (10,14), info.shape
            tool_boxes = info[:5, :3]
            tool_onehot = np.zeros((10,3))  # PourWater has 3 segm classes
            tool_onehot[:,0] = 1   # 1st column for one-hot is the tool

            # Add more tool points to avoid potential SVD instability. Take the
            # mean of tool points, then 4 more by taking values closer to bottom.
            alpha = 0.9
            tool_more = np.zeros((5,3))
            tool_more[0] = np.mean(tool_boxes, axis=0)  # center of tool points
            tool_more[1] = alpha * tool_boxes[0] + (1-alpha) * tool_boxes[1]
            tool_more[2] = alpha * tool_boxes[0] + (1-alpha) * tool_boxes[2]
            tool_more[3] = alpha * tool_boxes[0] + (1-alpha) * tool_boxes[3]
            tool_more[4] = alpha * tool_boxes[0] + (1-alpha) * tool_boxes[4]
            reduced_tool = np.vstack((tool_boxes, tool_more))

            # Form the 'top' of the point cloud.
            tool_segm = np.concatenate((reduced_tool, tool_onehot), axis=1)  # (10,6)

            # Again assume tool points come first. Remove old tool, add new tool.
            tool_idxs = np.where(obs[:,3] == 1)[0]
            obs_notool = obs[len(tool_idxs):]  # remove first `len(tool_idxs)` rows
            obs = np.concatenate((tool_segm,obs_notool), axis=0)  # stack tool
            return obs
        else:
            assert self.reduce_tool_PCL == 'None', self.reduce_tool_PCL


# ---------------------------------------------------------------------------------- #
# Some extra stuff for point cloud augmentation (potentially).
# ---------------------------------------------------------------------------------- #

def random_crop_pc(obs, action, max_x, min_x, max_y, min_y, max_z, min_z):
    """From Yufei, might be useful for data augmentation."""
    gripper_pos = obs.pos[obs.x[:, 2] == 1]

    gripper_x_min, gripper_x_max = torch.min(gripper_pos[:, 0]).item(), torch.max(gripper_pos[:, 0]).item()
    gripper_y_min, gripper_y_max = torch.min(gripper_pos[:, 1]).item(), torch.max(gripper_pos[:, 1]).item()
    gripper_z_min, gripper_z_max = torch.min(gripper_pos[:, 2]).item(), torch.max(gripper_pos[:, 2]).item()

    x_start = np.random.rand() * (gripper_x_min - min_x) + min_x
    y_start = np.random.rand() * (gripper_y_min - min_y) + min_y
    z_start = np.random.rand() * (gripper_z_min - min_z) + min_z

    x_end = x_start + (max_x - min_x) * 0.75
    y_end = y_start + (max_y - min_y) * 0.75
    z_end = z_start + (max_z - min_z) * 0.75

    x_end = max(x_end, gripper_x_max)
    y_end = max(y_end, gripper_y_max)
    z_end = max(z_end, gripper_z_max)

    mask = (obs.pos[:, 0] <= x_end) & (obs.pos[:, 0] >= x_start) & \
            (obs.pos[:, 1] <= y_end) & (obs.pos[:, 1] >= y_start) & \
            (obs.pos[:, 2] <= z_end) & (obs.pos[:, 2] >= z_start)

    obs.pos = obs.pos[mask]
    obs.x = obs.x[mask]
    obs.batch = obs.batch[mask]

    return obs, action[mask]


def rotate_pc(obs, angles=None, device=None, return_rot=False):
    """From Yufei, might be useful for data augmentation.

    Note: `obs` should be just positions, e.g., from a PCL.
    Tested on the real datasets, need to test more in sim.
    """
    if angles is None:
        angles = np.random.uniform(-np.pi, np.pi, size=3)
    Rx = np.array([[1,0,0],
                [0,np.cos(angles[0]),-np.sin(angles[0])],
                [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                [0,1,0],
                [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                [np.sin(angles[2]),np.cos(angles[2]),0],
                [0,0,1]])

    R = np.dot(Rz, np.dot(Ry,Rx))
    if device is not None:
        R = torch.from_numpy(R).to(device).float()
    obs = obs @ R   # (N,3) x (3x3)
    if return_rot:
        return (obs, R)
    else:
        return obs


def scale_pc(obs, scale_low, scale_high):
    """From Yufei, might be useful for data augmentation."""
    s = np.random.uniform(scale_low, scale_high)
    obs.pos = obs.pos * s
    return obs

# ---------------------------------------------------------------------------------- #
# Visualize the data
# ---------------------------------------------------------------------------------- #

def plot_action_hist_buffer(acts, scaling=False):
    """Plots histogram of the delta (x,y,z) translation values in data.

    The network should be predicting values in these ranges at test time.
    Also does rotations as well, assuming `acts.shape` is 6.

    Note: this will only work if our actions are interpreted as 6D translation
    and rotations, not if using flow.

    Values are mostly calibrated for the mixed media env.
    """
    return
    import matplotlib.pyplot as plt
    delta_x = acts[:,0]
    delta_y = acts[:,1]
    delta_z = acts[:,2]
    str_x = f'$\Delta$x, ({np.min(delta_x):0.4f}, {np.max(delta_x):0.4f})'
    str_y = f'$\Delta$y, ({np.min(delta_y):0.4f}, {np.max(delta_y):0.4f})'
    str_z = f'$\Delta$z, ({np.min(delta_z):0.4f}, {np.max(delta_z):0.4f})'
    l_x = f'x, {np.mean(delta_x):0.4f} +/- {np.std(delta_x):0.4f}'
    l_y = f'y, {np.mean(delta_y):0.4f} +/- {np.std(delta_y):0.4f}'
    l_z = f'z, {np.mean(delta_z):0.4f} +/- {np.std(delta_z):0.4f}'

    # Are we also plotting rotations?
    plot_rots = acts.shape[1] == 6
    if plot_rots:
        nrows, ncols = 2, 3
        drot_x = acts[:,3]
        drot_y = acts[:,4]
        drot_z = acts[:,5]
        str_drot_x = f'$\Delta$rotx, ({np.min(drot_x):0.4f}, {np.max(drot_x):0.4f})'
        str_drot_y = f'$\Delta$roty, ({np.min(drot_y):0.4f}, {np.max(drot_y):0.4f})'
        str_drot_z = f'$\Delta$rotz, ({np.min(drot_z):0.4f}, {np.max(drot_z):0.4f})'
        l_drot_x = f'x, {np.mean(drot_x):0.4f} +/- {np.std(drot_x):0.4f}'
        l_drot_y = f'y, {np.mean(drot_y):0.4f} +/- {np.std(drot_y):0.4f}'
        l_drot_z = f'z, {np.mean(drot_z):0.4f} +/- {np.std(drot_z):0.4f}'
    else:
        nrows, ncols = 1, 3

    # Scaling the ranges. Original ranges were set assuming that we had divided
    # by 0.004 (or multiplied by 250) so just undo that here.
    factor = 1.
    if not scaling:
        factor = 250

    # Bells and whistles.
    figname = f'fig_actions_scaling_{scaling}.png'
    titlesize = 32
    ticksize = 28
    legendsize = 23

    _, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(11*ncols, 7*nrows))
    ax[0,0].set_title(str_x, size=titlesize)
    ax[0,1].set_title(str_y, size=titlesize)
    ax[0,2].set_title(str_z, size=titlesize)
    ax[0,0].hist(delta_x, label=l_x, bins=20, edgecolor='k')
    ax[0,1].hist(delta_y, label=l_y, bins=20, edgecolor='k')
    ax[0,2].hist(delta_z, label=l_z, bins=20, edgecolor='k')
    ax[0,0].set_xlim([-1/factor, 1/factor])
    ax[0,1].set_xlim([-1/factor, 1/factor])
    ax[0,2].set_xlim([-1/factor, 1/factor])
    #ax[0,0].set_yscale('log')
    #ax[0,1].set_yscale('log')
    #ax[0,2].set_yscale('log')

    if plot_rots:
        ax[1,0].set_title(str_drot_x, size=titlesize)
        ax[1,1].set_title(str_drot_y, size=titlesize)
        ax[1,2].set_title(str_drot_z, size=titlesize)
        ax[1,0].hist(drot_x, label=l_drot_x, bins=20, edgecolor='k')
        ax[1,1].hist(drot_y, label=l_drot_y, bins=20, edgecolor='k')
        ax[1,2].hist(drot_z, label=l_drot_z, bins=20, edgecolor='k')
        ax[1,0].set_xlim([-1/factor, 1/factor])
        ax[1,1].set_xlim([-1/factor, 1/factor])
        ax[1,2].set_xlim([-1/factor, 1/factor])
        #ax[1,0].set_yscale('log')
        #ax[1,1].set_yscale('log')
        #ax[1,2].set_yscale('log')

    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    plt.savefig(figname)

    print(f'[replay buffer] See plot: {figname}')
