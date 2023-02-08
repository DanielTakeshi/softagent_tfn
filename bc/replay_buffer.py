"""Support two types of replay buffers

A 'traditional' buffer where point clouds (PCLs) have same # of points.
Another 'PC' buffer where we can have different # of points in each PCL.
"""
from macpath import basename
import sys
import os
from os.path import join
from operator import itemgetter
import numpy as np
import pickle
import torch
from torch_geometric.data import Data, Batch
from bc.bc import FLOW_ACTS, ALL_ACTS, PCL_MODELS
from bc import utils as U
import matplotlib.pyplot as plt
DEG_TO_RAD = np.pi / 180.


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
    Note: `acts` is optional if we also want to rotate translation.
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


class BehavioralCloningData():
    """Replay buffer to store data for Behavioral Cloning.

    Two subclasses for this depending on the nature of the point cloud data.
    If using images, keep values between 0 to 255 (encoder later divides by 255).

    Only need to save (obs,act) pairs and not rewards, but we should use a train
    / eval split. In `FlexEnv.reset()`, the last 20% of config indices are set as
    validation. Do something similar here by defining train / eval configs.
    """

    def __init__(self, args, obs_shape, action_shape, info_shape, device,
            action_lb=None, action_ub=None):
        self.args = args
        self.scale_pcl_flow = args.scale_pcl_flow
        self.scale_pcl_val = args.scale_pcl_val
        self.scale_targets = args.scale_targets
        self.bc_data_dir = args.bc_data_dir
        self.encoder_type = args.encoder_type
        self.batch_size = args.batch_size
        self.capacity = args.data_buffer_capacity
        self.image_size_crop = args.image_size_crop
        self.data_augm_PCL = args.data_augm_PCL
        self.obs_shape = obs_shape
        self.info_shape = info_shape
        self.device = device
        assert not (self.scale_pcl_flow and self.scale_targets), 'Do not use both'

        # Use the data-specific properties here.
        self.data_info = args.data_info
        self.n_train_demos = self.data_info['n_train_demos']
        self.n_valid_demos = self.data_info['n_valid_demos']
        self.max_train_demo = self.n_train_demos  # TODO(daniel) I think this is it?

        #! Again, using the VG argument directly. TODO(daniel) can we get rid of these?
        self.action_type = args.act_type  # the nature of what PN++ should predict
        self.action_lb = action_lb        # per-timestep act component lower bounds
        self.action_ub = action_ub        # per timestep act component upper bounds
        self.action_shape = action_shape

        # The proprioceptive obs is stored as float32, pixels obs as uint8.
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
        """New for BC, load from the data source.

        The exact type we load depends on the encoder, as we usually save many
        different types of observations for a fixed demonstrator (to keep things
        consistent). For now we've been saving observations as tuples:
            (keypts, img_rgb, segm_img, pc_array, flow)
        But need to keep this synced with SoftGym.

        If regressing to EE pose changes, re-scale actions to be in (-1,1) for ease
        of learning, otherwise values can be really small (up to +/- 0.004). Assumes
        symmetrical ranges, i.e., self.action_lb = -self.action_ub. If we disable,
        then check any time we query the BC policy (e.g., during evaluation). This
        also applies for tool flow, assuming it's in the same space as action deltas.

        If translation only in BC data, I'd expect flow to have similar min/max?
        BUT tool flow doesn't take action repeat into account. We have +/- 0.004 as
        defaults, so act repeat 8 means tool flow bounds are +/- 0.032, so divide
        by 8 here and it should now align with ee pose changes, so we can call
        `env.step(action)` consistently in external code.

        If using rotations, we do NOT scale these, and handle rotations elsewhere.
        """
        def get_obs_tool_flow(pcl, tool_flow):
            # NOTE(daniel): this is to get (obs,act) encoded correctly + consistently.
            # If `pcl` is segmented point cloud from time t-1, and `tool_flow`
            # is the flow from time t, then their tool points shoud coincide.
            # We will always provide (2000,d)-sized PCLs; in training, can resize.
            #! Since the point cloud in the physical demonstrations had the first objectness index as the target
            #! and the second as distractor and third as tool, its causing the assert below to mess up
            pcl_tool = pcl[:,3] == 1

            # print('[SARTHAK] pcl_shape: {} pcl_tool shape: {}'.format(pcl.shape, np.sum(pcl[:, 5] == 1)))
            tf_pts = tool_flow['points']
            tf_flow = tool_flow['flow']
            n_tool_pts_obs = np.sum(pcl_tool)
            n_tool_pts_flow = tf_pts.shape[0]
            # print('[SARTHAK] n_tool_pts_obs shape: {}, tf_pts shape: {}'.format(n_tool_pts_obs, tf_pts.shape))
            # First shapes only equal if: (a) fewer than max pts or (b) no item/distr.
            assert tf_pts.shape[0] <= pcl.shape[0], f'{tf_pts.shape}, {pcl.shape}'
            assert tf_pts.shape == tf_flow.shape, f'{tf_pts.shape}, {tf_flow.shape}'
            assert n_tool_pts_obs == n_tool_pts_flow, f'{n_tool_pts_obs}, {n_tool_pts_flow}'
            #! The softgym code assumes that the first n points are the toolpoints which is not the case
            #! in the physical experiment code
            assert np.array_equal(pcl[pcl_tool,:3], tf_pts)  # yay :)
            a = np.zeros((pcl.shape[0], 3))  # all non-tool point rows get 0s
            #! No need to encode flow in the way below, which is how softgym expects it.
            #! We can instead, directly check where the
            # ! a[:n_tool_pts_obs] = tf_flow   # actually encode flow for BC purposes
            a[pcl_tool, :] = tf_flow
            # print('[SARTHAK] a shape: {} pcl shape {}'.format(a.shape, pcl.shape))
            #! This is where we finally get state action  pairs from. pcl is the observation of the
            #! entire scene, whereas a is the action which implies points in the scene that were subjected
            #! to an action of some sorts which is the flow, applied only to those tool points
            return (pcl, a)

        # Load pickle paths into list. One item is one demonstration.
        print(f'\nLoading data for Behavioral Cloning: {self.bc_data_dir}')
        pkl_paths = sorted([
            join(self.bc_data_dir,x) for x in os.listdir(self.bc_data_dir)
                if x[-4:] == '.pkl' and 'BC' in x])

        # If filtering, load file which specifies config indices to keep. This is
        # later used in SoftGym since we'll have more configs and need to subsample.
        print(f'Loading {len(pkl_paths)} configs (i.e., episodes) from data.')
        self._filtered_config_idxs = [i for i in range(len(pkl_paths))]

        # Handle train and valid _config_ indexes (we only want filtered ones).
        self._train_config_idxs = self._filtered_config_idxs[:self.n_train_demos]
        self._valid_config_idxs = self._filtered_config_idxs[
                self.n_train_demos : self.n_train_demos + self.n_valid_demos]
        print(f'First {self.n_train_demos} idxs of starting configs are training.')
        print(f'Train configs (start,end), (inclusive,inclusive): '
            f'{self._train_config_idxs[0]}, {self._train_config_idxs[-1]}')
        print(f'Valid configs (start,end), (inclusive,inclusive): '
            f'{self._valid_config_idxs[0]}, {self._valid_config_idxs[-1]}')

        starting_obs_save = os.path.join(self.bc_data_dir, 'start_obs')

        os.makedirs(starting_obs_save)

        for vidx in range(self._valid_config_idxs[0], self._valid_config_idxs[-1]):
            with open(pkl_paths[vidx], 'rb') as fopen:
                data = pickle.load(fopen)
                print('Extracting vidx: {}'.format(basename(pkl_paths[vidx])))
                np.save(join(starting_obs_save, 'start_obs_{}.npy'.format(vidx)), data['obs'][0][3])

        # Action bounds. Careful about rotations.
        assert self.action_type in ALL_ACTS, self.action_type
        print(f'Action type: {self.action_type}. Act bounds:')
        print(f'  lower: {self.action_lb}')
        print(f'  upper: {self.action_ub}')

        # Iterate through filtered paths, only keeping what we need.
        for pidx,pkl_pth in enumerate(pkl_paths):

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
            len_a = len(data[act_key])
            assert len_o == len_a, f'{len_o} vs {len_a}'

            # Add each (obs,act) from this episode into the data buffer.
            # The `obs` is actually a tuple, so extract appropriate item.
            for t in range(len(data['obs'])):
                obs_tuple = data['obs'][t]
                act_raw = data[act_key][t]

                if self.encoder_type == 'pixel':
                    obs = np.transpose(obs_tuple[1], (2,0,1))
                elif self.encoder_type == 'segm':
                    obs = np.transpose(obs_tuple[2], (2,0,1))
                elif self.encoder_type in PCL_MODELS:
                    obs = obs_tuple[3]
                else:
                    raise NotImplementedError(self.encoder_type)

                if self.action_type == 'ee':
                    # Just use 3DoF actions.
                    act = act_raw[:3]
                elif self.action_type == 'eepose':
                    # Use 6DoF actions.
                    act = act_raw
                elif self.action_type == 'flow':
                    # In softgym, the flow is computed from the t-1 for t, whereas
                    # for the physical setup we post-process all of this and for a
                    # given time t's pointcloud, we have access to the flow as well
                    obs_tuple_next = data['obs'][t]
                    tool_flow = obs_tuple_next[4]
                    obs, act = get_obs_tool_flow(obs, tool_flow)
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
        print(f'  scaling targets? {self.scale_targets}')
        print(f'  Train idxs for (s,a) pairs (incl, excl): '
            f'({self.first_idx_train},{self.last_idx_train})')
        print(f'  Valid idxs for (s,a) pairs (incl, excl): '
            f'({self.first_idx_valid},{self.last_idx_valid})')
        self._debug_print()
        print()

        # Plot action data for further investigation.
        if isinstance(self.actions, np.ndarray):
            acts = self.actions[:self.idx]  # empty after self.idx
            U.plot_action_hist_buffer(acts, suffix=self.data_info['suffix'],
                    scaling=(self.scale_targets or self.scale_pcl_flow))

        # Optionally generate some visualizations with data augmentation?
        #self._test_data_augmentation()


class PointCloudReplayBuffer(BehavioralCloningData):
    """Replay buffer to store point cloud data (s,a) pairs.

    Big difference from the normal replay buffer: we allow for a variable
    number of points per PCL, assuming `self.remove_zeros_PCL = True`. (It
    probably should be True.)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.remove_zeros_PCL = self.args.remove_zeros_PCL

        # We can have PCL but not necessarily have actions in 2D arrays.
        self.use_PCL_act_list = self.action_type in ['flow', 'ee2flow']

        # Support a variable amount of points in point clouds (also for actions!).
        self.obses = []
        if self.use_PCL_act_list:
            self.actions = []
        else:
            self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.info = np.empty((self.capacity, *self.info_shape), dtype=np.float32)

        self._load_from_data()

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
        """
        if self.scale_pcl_flow:
            raise ValueError('Please do not do this')
            obs[:, :3] *= self.scale_pcl_val  # first 3 columns are xyz
            action *= self.scale_pcl_val  # applies for EE translations and flow
            info[:7] *= self.scale_pcl_val  # first 7 parts have position info
        elif self.scale_targets:
            # Scale transl. parts to roughly (-1,1), assumes symmetrical bounds!
            # Either action is just (3,) for EE or (N,3) for flow.
            if len(action.shape) == 1:
                #* This implies that the action is basically a vector that you're regressing to, like EE Pose or Flow2EE
                # print('before scaling applied ction[:3]'.format(action[:3]))
                action[:3] = action[:3] / self.action_ub[:3]
                # print('after scaling applied action[:3]'.format(action[:3]))
            else:
                action[:,:3] = action[:,:3] / self.action_ub[:3]

        if self.remove_zeros_PCL:
            # print('[SARTHAK] OBS Shape: {}'.format(obs.shape))
            # Find indices of various parts in the segmented point cloud.
            tool_idxs = np.where(obs[:,3] == 1)[0]
            targ_idxs = np.where(obs[:,4] == 1)[0]
            if obs.shape[1] == 6:
                # print('[SARTHAK] Yes there are distractors here')
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

        As of 06/11/2022 we support data augmentation, so there's a little bit of
        noise each time we sample the data. Since in real we aren't going to be
        scaling the point clouds (that was a special case in sim due to action
        repeat) we can use the same scale for all. For gaussian noise, encode it
        as a string: 'gaussian_variance'.
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
            assert actions.shape[0] == len(obses.ptr)-1, actions.shape

        if train:
            if 'rot_gaussian_' in self.data_augm_PCL:
                # Rotation augmentation. How to vectorize?
                for i in range(len(obses.ptr)-1):
                    idx1 = obses.ptr[i].detach().cpu().numpy().item()
                    idx2 = obses.ptr[i+1].detach().cpu().numpy().item()
                    obs_raw = torch.clone(obses.pos[idx1:idx2])  # PCL's xyz (tool+item).
                    tool_one = torch.where(obses.x[idx1:idx2, 0] == 1)[0]
                    tool_raw = obs_raw[tool_one]

                    # Or actually, do this at the _tip_ of the tool. Maximum z coord.
                    max_z = torch.where(tool_raw[:,2] == torch.max(tool_raw[:,2]))[0]
                    if len(max_z) > 1:
                        max_z = max_z[0]
                    obs_raw_mean = tool_raw[max_z]
                    obs_raw_cent = obs_raw - obs_raw_mean

                    # Rotate about the z-axis. NOTE(daniel): see our debug visualizations.
                    angle_deg = np.random.uniform(-180., 180.)
                    angle_rad = angle_deg * DEG_TO_RAD
                    angles = np.array([0, 0, angle_rad])
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
                    #fig_raw = U.pcl_data_aug_viz(
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

                # Additionally apply Gaussian noise as well? Should be in string form
                # like: 'rot_gaussian_xyz' so split and use the last index. Vectorized.
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

        if get_info:
            info = torch.as_tensor(self.info[idxs], device=self.device)
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
        num_pts = [data.x.shape[0] for data in self.obses]

        # Observations. Ball is at column 0 in data.x, unlike in simulation FYI.
        print(f'  #obs: {len(self.obses)}, each is a PCL of possibly different size')
        print(f'    data.x shape: {np.mean(num_pts):0.2f} +/- {np.std(num_pts):0.1f}')
        print(f'    #pts min max: {np.min(num_pts)}, {np.max(num_pts)}')
        num_pts_t = [len(np.where(data.x[:,0] == 1)[0]) for data in self.obses]
        num_pts_b = [len(np.where(data.x[:,1] == 1)[0]) for data in self.obses]
        print(f'    # tool pts: {np.mean(num_pts_t):0.2f} +/- {np.std(num_pts_t):0.1f}')
        print(f'    # ball pts: {np.mean(num_pts_b):0.2f} +/- {np.std(num_pts_b):0.1f}')
        print(f'    min max (tool): {np.min(num_pts_t)}, {np.max(num_pts_t)}')
        print(f'    min max (ball): {np.min(num_pts_b)}, {np.max(num_pts_b)}')

        # Actions
        if isinstance(self.actions, np.ndarray):
            print(f'  act: {self.actions.shape}, type {type(self.actions)}')
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

    def _test_data_augmentation(self):
        """Test the flow visualizations."""
        if False:
            return
        idx = 0
        flow_dir = 'tmp'

        # Just get positions. Note: careful if torch or numpy.
        obs_raw = (self.obses[idx].pos).to(self.device)
        pts_raw = (obs_raw).cpu().numpy()

        # # Zero-center the obs at the PCL mean.
        # obs_raw_mean = torch.mean(obs_raw, axis=0)
        # obs_raw_cent = obs_raw - obs_raw_mean

        # Or actually, do this at the _tip_ of the tool. Maximum z coord.
        max_z = torch.where(obs_raw[:,2] == torch.max(obs_raw[:,2]))[0]
        obs_raw_mean = obs_raw[max_z]
        obs_raw_cent = obs_raw - obs_raw_mean

        # Get augmented version. Clone since it modifies the obs.
        angle_deg = 45.
        angle_rad = angle_deg * DEG_TO_RAD
        angles = np.array([0, 0, angle_rad])
        obs_aug, R = rotate_pc(
                obs=torch.clone(obs_raw_cent),
                angles=angles,
                device=self.device,
                return_rot=True
        )
        R = R.cpu().numpy()

        # Back to the uncentered raw version.
        obs_aug = obs_aug + obs_raw_mean
        pts_aug = (obs_aug).cpu().numpy()

        # Handle flow vectors. If we have 3D actions (i.e., translation only) just
        # repeat the flow across the array.
        act = self.actions[idx]
        if act.shape == (3,):
            print(f'Flow viz for act: {act}')
            flow_raw = np.zeros_like(pts_raw)
            flow_raw += act
            flow_aug = np.zeros_like(pts_raw)
            flow_aug += act
            flow_aug = flow_aug @ R
        else:
            raise NotImplementedError()

        # Generate flow visualization as html and exit.
        fig_raw = U.pcl_data_aug_viz(pts_raw, pts_aug, flow_raw, flow_aug=flow_aug)
        viz_pth = join(flow_dir, f'flow_ang_{angles}.html')
        fig_raw.write_html(viz_pth)
        print(f'See visualization: {viz_pth}')
        sys.exit()
