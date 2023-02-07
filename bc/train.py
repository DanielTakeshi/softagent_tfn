import pickle
import numpy as np
import torch
import os
import time
import json
import cv2
from os.path import join
from collections import defaultdict
from scipy.spatial.transform import Rotation as Rot
from bc.utils import MixedMediaToolReducer
from bc.env_specific_hacks import (
    get_env_act_bounds, get_env_action_mask, check_env_version, env_scaling_targets,
    get_tool_origin_testtime, get_info_shape, get_state_dim
)

np.set_printoptions(edgeitems=20)
torch.set_printoptions(edgeitems=20)

# NOTE(daniel): to reduce error prone code, try and use constants from `bc.bc`.
from bc import utils, rotations
from bc import replay_buffer
from bc.bc import (
    BCAgent, PIXEL_ENC, PCL_MODELS, FLOW_ACTS, ALL_ACTS, ENV_ACT_MODES, PCL_COMPRESS,
    AVOID_FLOW_PREDS, PCL_OBS, ROT_REPRESENTATIONS, NO_RPMG_REPRESENTATIONS
)
from bc.pointnet2_classification import PointNet2_Class
from bc.logger import Logger
from bc.utils import update_env_kwargs
from chester import logger
from envs.env import Env, SoftGymEnv
from softgym.utils.visualization import (
    save_numpy_as_gif, make_grid, save_pointclouds
)
import matplotlib.pyplot as plt
import wandb
from pytorch3d.transforms import quaternion_to_axis_angle
from pyquaternion import Quaternion


def train_state_encoder(net, data, batch_size=24, num_epochs=100):
    """Train the state encoder (Phase 1).

    Empirically, 100 epochs makes validation loss stabilize at ~0.0002 for scooping.
    """
    import torch.nn as nn
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    MSE_loss = nn.MSELoss()

    # Get # of minibatches for train and validation.
    n_train_items = data.num_train_items()
    min_i, max_i = data.get_valid_idx_ranges()
    n_train_mbs = int(n_train_items / batch_size)
    n_valid_mbs = int((max_i - min_i) / batch_size)
    print('\nNow training the state encoder. Data info:')
    print(f'  train items: {n_train_items}, minibatches {n_train_mbs}')
    print(f'  valid items: {max_i-min_i}, minibatches {n_valid_mbs}')
    time_s = time.time()

    for i in range(num_epochs):
        print(f'Train epoch {str(i+1).zfill(3)} / {num_epochs}')

        # Evaluate
        net.eval()
        with torch.no_grad():
            v_losses = []
            for vv in range(n_valid_mbs):
                vmin = min_i + (batch_size * vv)
                vmax = min_i + (batch_size * (vv+1))
                obs_valid, state_gt = data.sample_obs_act(
                        train=False, v_min_i=vmin, v_max_i=vmax)
                state_pred = net(obs_valid)
                v_loss = MSE_loss(state_pred, state_gt)
                v_losses.append(v_loss.item())
            print(f'  valid loss: {np.mean(v_losses):0.5f}')

        # Train
        net.train()
        t_losses = []
        for _ in range(n_train_mbs):
            obs, state_gt = data.sample_obs_act(get_info=False)
            state_pred = net(obs)
            t_loss = MSE_loss(state_pred, state_gt)
            t_losses.append(t_loss.item())
            optim.zero_grad()
            t_loss.backward()
            optim.step()
        print(f'  train loss: {np.mean(t_losses):0.5f}')

    net.eval()
    time_e = time.time() - time_s
    print(f'Done with training PCL --> state encoder, time: {time_e:0.3f}')


def vv_to_args(vv):
    class VArgs(object):
        def __init__(self, vv):
            for key, val in vv.items():
                setattr(self, key, val)

    args = VArgs(vv)

    # Dump parameters. NOTE(daniel) careful! Later in `main()` we update these,
    # so look at anything starting with `env_kwargs_` such as the obs mode which
    # might override stuff in the _dict_ with name `env_kwargs` in the args.
    variant_path = os.path.join(logger.get_dir(), 'variant.json')
    with open(variant_path, 'w') as f:
        json.dump(vv, f, indent=2, sort_keys=True)
    wandb.save(variant_path)

    return args


def run_task(vv, log_dir=None, exp_name=None):
    if log_dir or logger.get_dir() is None:
        logger.configure(dir=log_dir, exp_name=exp_name, format_strs=['csv', 'wandb'], variant=vv)
    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)
    updated_vv = {}  # NOTE(daniel): previously we had a 'default' dict to start with.
    updated_vv.update(**vv)  # NOTE(daniel): adds SoftGym arguments.
    main(vv_to_args(updated_vv))


def get_info_stats(infos):
    """
    NOTE(daniel) from this I think we really should keep the length of each episode the
    same, so no early termination. Forms `info_*_mean` and `info_*_final` stats for plots.
    """
    # infos is a list with N_traj x T entries
    N = len(infos)
    T = len(infos[0])
    stat_dict_all = {key: np.empty([N, T], dtype=np.float32) for key in infos[0][0].keys()}
    for i, info_ep in enumerate(infos):
        for j, info in enumerate(info_ep):
            for key, val in info.items():
                stat_dict_all[key][i, j] = val

    stat_dict = {}
    for key in infos[0][0].keys():
        # stat_dict_all[key] should be array of shape (N_traj x T)
        stat_dict[key + '_mean'] = np.mean(np.array(stat_dict_all[key]))
        stat_dict[key + '_final'] = np.mean(stat_dict_all[key][:, -1])
    return stat_dict


def action_conversion(env, action, args, record_acts, debug=False):
    """How do we handle actions for BC?

    It's confusing since the env.step(...) here normalizes the action, so we can
    anticipate that and un-normalize. But we might also scale the actions ourselves
    to make BC predict 'reasonably-sized' values. Originally actions are implemented
    as changes bounded by (-0.004, 0.004) in xyz each, but we re-scale to (-1,1) for
    BC predictions, so whatever BC predicts, multiply by 0.004 to down-scale it. If
    we change this, check the data buffer since that's where we amplified the labels.

    For tool flow, values were specified in (-0.004*ar, 0.004*ar) where ar is the
    action repeat, and then we rescaled to (-1,1) per component to be in the buffer.
    This means it's within the same per-component scaling as regressing to a single
    action, so we just need to output by 0.004.

    Rotations in axis-angle formulation (for now) use bounds of +/-1 so
    those should be the same for the scaled and unscaled variants.

    #lb = env.action_space.low  # this is (-1,-1,-1), don't use
    #ub = env.action_space.high  # this is (1,1,1), don't use
    #action_repeat = env._env.action_repeat  # this is 8 but we don't need here.

    04/21/2022: need to now consider (identity) quaternions due to SVD layer.
    04/25/2022: now consider non-identity quaternions (the general case).
    05/08/2022: cleaning up to handle more cases, etc.
    07/11/2022: slightly clean up the 'action projection'.

    Parameters
    ----------
    action: The action as outputted by the BC policy network.
    args: From VariantGenerator.
    record_acts: dict where we can populate the raw and the processed acts,
        useful if debugging later (e.g., loading trained BC policies).
    """

    # HACK! TODO(daniel) must remove. This probably happens if there exist
    # no tool points in the point cloud. For training, there should always
    # be such tool points in the point cloud by how we generated the data.
    # Get rid of this by always ensuring there exists tool points in PCL.
    if np.isnan(np.sum(action)):
        print(f'WARNING! action: {action}. We will override but please fix!')
        action = np.zeros_like(action)

    record_acts['from_pol'].append(action)
    assert args.env_kwargs['action_mode'] in ENV_ACT_MODES, args.env_kwargs
    assert args.act_type in ALL_ACTS, args.act_type
    assert len(action) in [3,6,9,12,13], action

    # # Handle quaternions if our policy outputted that directly.
    # # TODO(daniel) this is deprecated, commenting out now, removing later!
    # if len(action) == 7:
    #     assert args.env_kwargs['action_mode'] == 'translation_axis_angle'
    #     assert args.use_geodesic_dist
    #     act_q = quaternion_to_axis_angle(torch.as_tensor(action[3:]))
    #     action = np.concatenate( (action[:3], act_q.numpy()) )

    # NOTE(daniel): another hack, we need to fix these special cases. As of 06/21,
    # this case handles when we use flow methods to predict 6D actions, but where
    # the env is translation-only (so we ignore the last 3 components).
    if args.env_kwargs['action_mode'] == 'translation':
        if len(action) == 6:
            action = action[:3]

    # If we scaled targets, then move BC transl. preds. back to (-0.004,0.004)
    # per component, as that's what we used to define the algo policy.
    lb, ub = get_env_act_bounds(env)
    if args.scale_targets:
        if debug:
            print(f'w/{action}, call env.step({action * ub})')
        action = env_scaling_targets(
            act=action,
            act_ub=ub,
            env_name=args.env_name,
            env_version=args.env_version,
            inference=True
        )
    elif args.scale_pcl_flow:
        # TODO(daniel) is this valid? By scaling, we did all computations
        # assuming that PCLs and flow are expressed in mm (instead of m). Then
        # we got (R,t) from that. The t seems like it should be re-scaled to m,
        # which dividing here will do, but does the rotation stay the same?
        # I _think_ it does after doing some toy examples, but please confirm.
        if debug:
            tmp = action[:3] / args.scale_pcl_val
            print(f'w/{action}, call env.step({tmp})')
        # Should also work if we have translation-only data w/flow methods.
        action[:3] = action[:3] / args.scale_pcl_val

    # More debugging / recording. Assumes symmetrical (-x,x) delta constraints.
    record_acts['from_pol_times_ub'].append(action)
    record_acts['exceed_dx'].append(np.abs(action[0]) > ub[0])
    record_acts['exceed_dy'].append(np.abs(action[1]) > ub[1])
    record_acts['exceed_dz'].append(np.abs(action[2]) > ub[2])

    # Convert SVD's global axis into a local axis for softgym. NOTE(daniel): ONLY
    # for MMOneSphere or MMMultiSphere as other envs might not have `env.tool_state`.
    # Also this should only be for something that used SVD, right? Not direct vector?
    # convert_axes = (
    #     (args.method_flow2act == 'svd' or '_pointwise' in args.encoder_type) and
    #     (args.encoder_type not in ['pointnet_svd_6d_flow_mse_loss', 'pointnet_svd'])
    # )

    # Convert if we are doing anything other than directly regressing to the environment action
    convert_axes = (args.act_type != "eepose")

    if len(action) != 3 and convert_axes:
        action = rotations.deconvert_action(
            # This needs to work with flow when we aren't testing rotation representations
            args.rotation_representation if (args.rotation_representation in ROT_REPRESENTATIONS or args.act_type == "eepose_convert") else "flow",
            args.env_name,
            action,
            env
        )

    # Anticipate the env's normalization so `action` has intended meaning. When
    # we do `env.step(denorm_action)`, this makes `denorm_action` -> `action`.
    denorm_action = (action - lb) / ((ub - lb) * 0.5) - 1.0
    if debug:
        print(f'env.step({denorm_action}) --> {action}, lb,ub: {lb} {ub}')

    # Important! Apply env-specific projections to go from 6D -> (lower D).
    if args.project_axis_ang_y or args.project_pour_water:
        assert len(denorm_action) == 6, denorm_action
        _mask = get_env_action_mask(env_name=args.env_name, env_version=args.env_version)
        denorm_action = denorm_action * _mask
    record_acts['denorm_action'].append(denorm_action)

    return denorm_action


def process_test_time_obs(args, obs, env, tool_reducer):
    """Process test-time observations.

    We handle: (a) clearing out zeros in point clouds, (b) scaling the obs
    point cloud, and (c) potentially the tool origin (tip for MM) as well.
    Interestingly, this works with np.where even though `obs` is a torch
    tensor? https://github.com/pytorch/pytorch/issues/59237 Also called
    even if using images but should not do anything to `obs`.

    05/21/2022: now handle dense transformation where we have a new point
    which is a synthetic point on the point cloud representing the tool tip.
    For this, treat that as a tool point? DESIGN CHOICE. Putting it as a
    separate class entirely seems like it could make it a lot harder...also
    should be OK to just override the 1st point (kind of like data aug).

    06/01/2022: support PourWater. I think we want the tool origin which is
    the env's (x,y). Also the zero filtering and dense transforms should be
    OK (obs.shape is 6) but careful if this changes.

    06/05/2022: support noise injection, using the same noise level as in
    training data (handled in the replay buffer). Do this before scaling.

    06/08/2022: support reducing tool point cloud. Of course, a similar
    procedure must be done for training observations.

    Returns
    -------
    (obs, tool_coord_center): The processed observation and the center of
        the tool's coordinate frame. In external code this is called the
        `tool_tip_pos` since the tip of the ladle acted as its origin.
    """
    tool_origin = get_tool_origin_testtime(env=env, env_name=args.env_name)

    # Double check images. For RGBD I think it's easier to keep as floats.
    if args.env_kwargs['observation_mode'] == 'cam_rgb':
        assert obs.dtype == torch.uint8, obs.dtype
    elif args.env_kwargs['observation_mode'] in ['cam_rgbd', 'depth_segm',
            'depth_img', 'rgb_segm_masks', 'rgbd_segm_masks', 'state']:
        assert obs.dtype == torch.float32, obs.dtype

    # Only do the following if using point cloud observations. This WILL include
    # the state_predictor_then_mlp case. We need to process PCLs the same way.
    if (args.env_kwargs['observation_mode'] not in PCL_OBS):
        return (obs, tool_origin)

    # Handle tool point cloud reduction (we don't have it as an observation type).
    if args.reduce_tool_PCL == 'pouring_v01':
        box_info = env.get_boxes_info()
        g_1 = box_info['glass_floor']
        g_2 = box_info['glass_left']
        g_3 = box_info['glass_right']
        g_4 = box_info['glass_back']
        g_5 = box_info['glass_front']
        assert np.abs(g_1[0] - env.glass_x) < 1e-5, \
                f'{g_1}, {env.glass_x},  {g_1[0] - env.glass_x}'
        assert np.abs(g_1[1] - env.glass_y) < 1e-5, \
                f'{g_1}, {env.glass_y},  {g_1[1] - env.glass_y}'
        tool_boxes = np.concatenate([
            g_1[:3][None,:],
            g_2[:3][None,:],
            g_3[:3][None,:],
            g_4[:3][None,:],
            g_5[:3][None,:],
        ], axis=0)  # (5,3)

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

        tool_onehot = np.zeros((10,3))  # PourWater has 3 segm classes
        tool_onehot[:,0] = 1.   # 1st column for one-hot is the tool
        tool_segm = np.concatenate((reduced_tool, tool_onehot), axis=1)  # (10,6)

        # Again assume tool points come first. Remove old tool, add new tool.
        tool_idxs = np.where(obs[:,3] == 1)[0]
        obs_notool = obs[len(tool_idxs):]  # remove first `len(tool_idxs)` rows
        obs = np.concatenate((tool_segm,obs_notool), axis=0)
        obs = torch.from_numpy(obs).float()
    elif tool_reducer is not None:
        tool_reducer.set_axis(env.tool_state[0, 6:10])
        # TODO(eddieli): Use ground truth env rotation data instead of just dead
        # reckoning
        obs = tool_reducer.reduce_tool(obs.detach().numpy(), info=env.tool_state[0, :3])
        obs = torch.from_numpy(obs).float()
    else:
        assert args.reduce_tool_PCL == 'None', args.reduce_tool_PCL

    # Since the env has given zero-padded point clouds, remove zeros.
    if args.remove_zeros_PCL:
        tool_idxs = np.where(obs[:,3] == 1)[0]
        targ_idxs = np.where(obs[:,4] == 1)[0]
        if obs.shape[1] == 6:
            # PourWater = water points, MM = distractor ball.
            dist_idxs = np.where(obs[:,5] == 1)[0]
        else:
            dist_idxs = np.array([])
        n_nonzero_pts = len(tool_idxs) + len(targ_idxs) + len(dist_idxs)
        if n_nonzero_pts < obs.shape[0]:
            nonzero_idxs = np.concatenate((tool_idxs, targ_idxs, dist_idxs))
            nonzero_idxs = torch.as_tensor(nonzero_idxs, dtype=torch.int64)
            obs = obs[nonzero_idxs]

    # First 3 columns are xyz positions in meters, usually quite small.
    if args.gaussian_noise_PCL > 0:
        obs[:, :3] += np.random.normal(0,
                args.gaussian_noise_PCL, size=obs[:,:3].shape)

    # Later the PN++ will actually downscale to make the inputs manageable.
    if args.scale_pcl_flow:
        obs[:, :3] *= args.scale_pcl_val
        tool_origin *= args.scale_pcl_val

    # Add synthetic point to 1st row of PCL, overriding existing point.
    # NOTE(daniel): slightly different from replay buffer code as we have
    # a fixed shape for tool_origin (if PourWater, adds the 0.).
    if args.dense_transform:
        fake_point = [tool_origin[0], tool_origin[1], tool_origin[2], 1., 0.]
        if obs.shape[1] == 6:
            fake_point.append(0.)
        fake_point_pt = torch.as_tensor(np.array(fake_point)).float()
        obs[0] = fake_point_pt

    # Last processing step before PyGeom, zero-center PCL individually.
    if args.zero_center_PCL:
        centroid_mean = torch.mean(obs[:,:3], dim=0)
        obs[:,:3] = obs[:,:3] - centroid_mean

    return (obs, tool_origin)


def evaluate(env, agent, video_dir, L, epoch, args, data_buffer, tool_reducer):
    """Run evaluation step, this is the 'eval' in the logger.

    See documentation in the corresponding CURL / SAC code. We support extra
    features relevant to BC, such as computing validation MSE.
    """
    all_ep_rewards = []
    num_episodes = args.n_eval_episodes
    num_eval_GIFs = min(25, num_episodes)
    num_rows_GIFs = 2 if num_eval_GIFs % 2 == 0 else 1
    if num_eval_GIFs > 10 and not args.test_overfitting:
        num_rows_GIFs = int(num_eval_GIFs / 5)

    # NOTE: the native pixel resolution for point clouds is 128. Unfortunately if
    # we change this we only scale up the image so it won't look as clean.
    pix = 128
    if args.load_model:
        pix = 256

    if args.test_overfitting:
        # Test overfitting to just the 0-th (or a particular) config. Careful,
        # these configs are the RAW ones and might be removed from filtering.
        # Check BC data to see if we are using one in training.
        num_episodes = 2
        num_eval_GIFs = 2
        overfit_cfg = 0  # Possibly use 1 for the 3DoF MMOneShere

    def run_eval_loop():
        start_time = time.time()
        infos = []
        all_frames = []
        all_pcls = []
        init_steps = []
        all_obs = []
        cfg_IDs = []
        success_eps = []
        plt.figure()

        for i in range(num_episodes):
            if args.load_model:
                print(f'Loading model {i} of {num_episodes}')

            # If overfitting, only want specific configs (those during training).
            if args.test_overfitting:
                obs = env.reset(config_id=overfit_cfg)
            elif args.test_each_valid_once:
                _filtered_valid = env.get_filtered_configs_valid_list()
                obs = env.reset(config_id=_filtered_valid[i])
            else:
                obs = env.reset()
            obs, tool_tip_pos = process_test_time_obs(args, obs, env=env, tool_reducer=tool_reducer)

            cfg_IDs.append(env.get_current_config_id())
            done = False
            info = None
            episode_reward = 0
            ep_info = []
            frames = [env.get_image(pix, pix)]
            pcls = [obs]  # use later if using PCLs
            rewards = []
            init_s = 0
            record_acts = defaultdict(list)

            while not done:
                # center crop image, if RGB: (1, 3, 128, 128) --> (3, 100, 100)
                # if segm: (1, n_classes, 128, 128) --> (n_classes, 100, 100)
                if args.encoder_type in PIXEL_ENC:
                    obs = utils.center_crop_image(obs, args.image_size_crop)
                all_obs.append((obs, tool_tip_pos))

                # Forward pass, then convert action if needed.
                action = agent.select_action(obs, info=tool_tip_pos)
                action_for_env = action_conversion(env, action, args, record_acts)
                obs, reward, done, info = env.step(action_for_env)
                obs, tool_tip_pos = process_test_time_obs(args, obs, env=env, tool_reducer=tool_reducer)
                # ------------------- back to normal ------------------- #
                episode_reward += reward
                ep_info.append(info)
                frames.append(env.get_image(pix, pix))
                pcls.append(obs)
                rewards.append(reward)

            # Special to early termination case. Freeze the GIFs, etc.
            if args.env_name == 'SpheresLadle' and len(frames) < 101:
                _tmp_frame = frames[-1]
                while len(frames) < 101:
                    frames.append(_tmp_frame)
                _tmp_info = ep_info[-1]
                while len(ep_info) < 100:
                    ep_info.append(_tmp_info)

            plt.plot(range(len(rewards)), rewards, label=f'Epis. {i}')
            if len(all_frames) < num_eval_GIFs:
                all_frames.append(frames)
                all_pcls.append(pcls)
                init_steps.append(init_s)
            infos.append(ep_info)
            success_eps.append( info['done']==1 )  # binary success
            L.log('eval/episode_reward', episode_reward, epoch)
            all_ep_rewards.append(episode_reward)

            # Mostly if we want to analyze / debug the loaded model.
            if args.load_model and False:
                print('\nfrom pol | from pol times ub | denorm_action')
                num_exceed_delta = 0.0
                exceeds_r = []
                exceeds_d = []
                for tt,(a1,a2,a3) in enumerate(zip(
                        record_acts['from_pol'],
                        record_acts['from_pol_times_ub'],
                        record_acts['denorm_action'])):
                    exceed = ''
                    # This is the magnitude assuming no clearing of x and z.
                    magn = np.linalg.norm(record_acts['from_pol_times_ub'][tt][3:])
                    magn_no_xz = np.linalg.norm(record_acts['denorm_action'][tt][3:])
                    if record_acts['exceed_dx'][tt]: exceed += ' x'
                    if record_acts['exceed_dy'][tt]: exceed += ' y'
                    if record_acts['exceed_dz'][tt]: exceed += ' z'
                    print(f'{str(tt).zfill(2)}, {a1}  {a2}  {a3}  magn {magn:0.3f} magn_no_xz {magn_no_xz:0.3f} {exceed}')
                    if magn > env.max_rot_axis_ang:
                        exceeds_r.append(magn - env.max_rot_axis_ang)
                        exceeds_d.append((magn - env.max_rot_axis_ang)*(180./np.pi))
                print('  exceed dx: {}'.format(np.sum(record_acts['exceed_dx'])))
                print('  exceed dy: {}'.format(np.sum(record_acts['exceed_dy'])))
                print('  exceed dz: {}'.format(np.sum(record_acts['exceed_dz'])))
                print('  avg pol: {}'.format(
                        np.mean(np.array(record_acts['from_pol']), axis=0)))
                print('  (absval): {}'.format(
                        np.mean(np.abs(np.array(record_acts['from_pol'])), axis=0)))
                print('  len exceeds: {}'.format(len(exceeds_r)))
                exceeds_r = np.array(exceeds_r)
                exceeds_d = np.array(exceeds_d)
                print('  avg exceed r: {}, {:0.3f} +/- {:0.1f}'.format(
                        exceeds_r, np.mean(exceeds_r), np.std(exceeds_d)))
                print('  avg exceed d: {}, {:0.3f} +/- {:0.1f}'.format(
                        exceeds_d, np.mean(exceeds_d), np.std(exceeds_d)))

            # Another way of analyzing to get nice policy rollouts.
            if args.load_model:
                succeed = info['done'] == 1
                dir_images = join(video_dir, f'ep_{i}_cam_rgb_{succeed}')
                dir_flows = join(video_dir, f'ep_{i}_flows_{succeed}')
                dir_data = join(video_dir, f'ep_{i}_raw_data_{succeed}')
                utils.make_dir(dir_images)
                utils.make_dir(dir_flows)
                utils.make_dir(dir_data)
                EPLEN = len(frames)

                # Only save flow visualizations if the episode was successful.
                # Requires that the agent is actually a flow-based one! Save
                # as html files so that we can rotate and adjust as needed.
                # Note that all_obs contains ALL obs from all episodes. Also
                # save the raw data so we can stitch everything for the website.
                # (Update Nov 2022: let's just always save except if NaN)
                if succeed or (not np.isnan(action).all()):
                    _just_these_obs = all_obs[-EPLEN:]
                    for fidx,(_obs, __info) in enumerate(_just_these_obs):
                        flow_dict = agent.get_single_flow(_obs, __info)
                        xyz = flow_dict['xyz'].detach().cpu().numpy()
                        flow = flow_dict['flow'].detach().cpu().numpy()
                        if 'flow_r' not in flow_dict:
                            fig = utils.create_flow_plot(xyz, flow, args=args)
                        else:
                            act = flow_dict['act'].detach().cpu().numpy()
                            flow_r = flow_dict['flow_r'].detach().cpu().numpy()
                            fig = utils.create_6d_flow_plot(xyz, flow, flow_r, act,
                                    sizeref=20, args=args, just_combined=True)
                        path_flow = join(dir_flows, f'flow_{str(fidx).zfill(3)}.html')
                        fig.write_html(path_flow)

                        # Save the raw data so we can plot later.
                        data_timestep = {
                            'obs': _obs,  # (N,6) point cloud, with M tool points
                            'xyz': xyz,  # (M,3), just tool points
                            'flow': flow,  # (M,3), tool point flow
                        }
                        path_raw_data = join(
                            dir_data, f'raw_data_{str(fidx).zfill(3)}_{xyz.shape[0]}_tool_pts.pkl'
                        )
                        with open(path_raw_data, 'wb') as fh:
                            pickle.dump(data_timestep, fh)

                # Save images.
                for (fr_idx, fr_rgb) in enumerate(frames):
                    dir_img = join(dir_images, f'rgb_{str(fr_idx).zfill(3)}.png')
                    cv2.imwrite(dir_img, cv2.cvtColor(fr_rgb, cv2.COLOR_RGB2BGR))
                print(f'See saved images: {dir_images}')

        # NOTE: this assumes initializer takes equal times per episode!
        n_init_steps = int(np.mean(init_steps))

        # Bells and whistles. Makes figure of eval performance.
        plt.title(f'Evaluation. Train Epochs: {epoch}', fontsize=16)
        plt.xlabel('Reward', fontsize=14)
        plt.ylabel('Time Step', fontsize=14)
        plt.legend(loc='best', ncol=2)
        plt.savefig(os.path.join(video_dir, f'{str(epoch).zfill(7)}.png'))

        # Save GIFs of RGB images. GIFs now have color change at initializer.
        all_frames = np.array(all_frames).swapaxes(0,1)  # shaped to: (time,#eps,128,128,3)
        subsamp = 1
        if args.env_kwargs['horizon'] > 200:
            subsamp = int(args.env_kwargs['horizon'] / 100)
        all_f = []
        for ft,frame in enumerate(all_frames):
            if ft % subsamp != 0:
                continue
            pv = 0 if ft < n_init_steps else 120
            all_f.append( make_grid(np.array(frame), nrow=num_rows_GIFs, padding=6, pad_value=pv) )
        all_frames = np.array(all_f)
        fname_10 = os.path.join(video_dir, f'{str(epoch).zfill(7)}_fps10.gif')
        save_numpy_as_gif(all_frames, filename=fname_10, fps=10)

        # Save the config IDs so we can check exact indices, and performance.
        cfg_IDs_pth = os.path.join(video_dir, f'{str(epoch).zfill(7)}_cfgs.txt')
        succ_IDs_pth = os.path.join(video_dir, f'{str(epoch).zfill(7)}_success.txt')
        with open(cfg_IDs_pth, 'w') as tf:
            for _cfg_id in cfg_IDs:
                print(f'{_cfg_id}', file=tf)
        with open(succ_IDs_pth, 'w') as tf:
            for _succ in success_eps:
                print(f'{_succ}', file=tf)

        # Save GIFs of point clouds if we use these as observations (slow!).
        if args.save_eval_PCLs and args.encoder_type in PCL_MODELS:
            all_pcls_np = []
            for obs_p in all_pcls:
                pcl_np = save_pointclouds(obs_p, n_views=1, return_np_array=True)
                all_pcls_np.append(pcl_np)  # pcl_np.shape: (time,height,width,3)
            all_pcls_np = np.array(all_pcls_np).swapaxes(0,1)  # shaped to: (time,#eps,H,W,3)
            all_p = []
            for ft,pcl in enumerate(all_pcls_np):
                pv = 0 if ft < n_init_steps else 120
                all_p.append( make_grid(pcl, nrow=num_rows_GIFs, padding=6, pad_value=pv) )
            all_pcls = np.array(all_p)
            fname_pcl = os.path.join(video_dir, f'{str(epoch).zfill(7)}_pcl.gif')
            save_numpy_as_gif(all_pcls, filename=fname_pcl, fps=10)

        # Dump info to logs.
        for key, val in get_info_stats(infos).items():
            L.log('eval/info_' + key, val, epoch)
        L.log('eval/eval_time', time.time() - start_time, epoch)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/mean_episode_reward', mean_ep_reward, epoch)
        L.log('eval/best_episode_reward', best_ep_reward, epoch)

        # Visualize random observation. These PCLs are NOT scaled, FYI.
        if ((agent.act_type in FLOW_ACTS or agent.encoder_type in PCL_COMPRESS)
                and agent.encoder_type not in AVOID_FLOW_PREDS):
            # Can pick random one, or just pick one early in training.
            obs, _info = all_obs[np.random.randint(0, len(all_obs))]
            #obs, _info = all_obs[np.random.randint(0, 10)]
            with torch.no_grad():
                flow_dict = agent.get_single_flow(obs, _info)
            xyz = flow_dict['xyz'].detach().cpu().numpy()
            flow = flow_dict['flow'].detach().cpu().numpy()
            if 'flow_r' not in flow_dict:
                # 3D flow case
                fig = utils.create_flow_plot(xyz, flow, args=args)
            else:
                # 6D flow case
                act = flow_dict['act'].detach().cpu().numpy()
                flow_r = flow_dict['flow_r'].detach().cpu().numpy()
                fig = utils.create_6d_flow_plot(xyz, flow, flow_r, act, args=args)
            logger.logkv('eval/flow_episode_plot', fig)

    def run_eval_mse():
        # Check valid MSE to see if it is generalizing to unseen data.
        min_i, max_i = data_buffer.get_valid_idx_ranges()
        n_valid_mbs = int((max_i - min_i) / args.batch_size)
        v_losses = []
        v_losses_pos = []
        v_losses_rot = []
        for vv in range(n_valid_mbs):
            vmin = min_i + (args.batch_size * vv)
            vmax = min_i + (args.batch_size * (vv+1))
            obs_valid, act_valid, info_valid = data_buffer.sample_obs_act(
                    train=False, v_min_i=vmin, v_max_i=vmax, get_info=True)
            v_loss, v_loss_p, v_loss_q = agent.evaluate_mse(
                    obs=obs_valid, act_gt=act_valid, info=info_valid)
            v_losses.append(v_loss)
            v_losses_pos.append(v_loss_p)
            v_losses_rot.append(v_loss_q)
        L.log('eval/MSE_loss', np.mean(v_losses), epoch)
        L.log('eval/MSE_loss_pos', np.mean(v_losses_pos), epoch)
        L.log('eval/MSE_loss_rot', np.mean(v_losses_rot), epoch)

    # Do MSE each time (if not overfitting) but maybe do rollouts less often.
    if epoch % args.eval_interval == 0:
        if isinstance(env, SoftGymEnv):
            env.set_env_eval_flag(True)
        run_eval_loop()
        if isinstance(env, SoftGymEnv):
            env.set_env_eval_flag(False)

    # If we load a model we might want to skip (or it could be good to verify...).
    if (not args.test_overfitting) and (not args.load_model):
        run_eval_mse()


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'bc':
        return BCAgent(
            args=args,
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            encoder_type=args.encoder_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def check_names(exp_name, load_dir):
    """Error checks to help ensure loading works."""
    exp_split = exp_name.split('_')
    load_split = os.path.basename(load_dir).split('_')
    assert exp_split[0] == load_split[0], f'{exp_split}, {load_split}'
    assert exp_split[1] == load_split[1], f'{exp_split}, {load_split}'
    assert exp_split[2] == load_split[2], f'{exp_split}, {load_split}'
    assert 'model' not in load_dir and 'buffer' not in load_dir, load_dir


def main(args):
    """Runs Behavioral Cloning.

    Creates `Env` here and if there are no cached configs, generate them. The `Env`
    is a wrapper around the usual normalized env use for quick SoftGym tests. The
    `action_repeat=1` but this is for `Env.step()`, we still use action repeat for
    the internal FlexEnv as specified in env_kwargs.

    Note: if using the state_predictor_then_mlp let's assume our encoder is going
    to be the PointNet++ because that encodes the PCL into a state target...
    """
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    # NOTE(daniel) if `symbolic` is False, then `Env` class does image processing,
    # (e.g., resizing, type checks) so avoid if using point_cloud or flow obs data.
    args.__dict__ = update_env_kwargs(args.__dict__)  # Update env_kwargs
    symbolic = True
    if args.env_kwargs['observation_mode'] in ['cam_rgb', 'cam_rgbd', 'depth_segm',
            'depth_img', 'rgb_segm_masks', 'rgbd_segm_masks']:
        assert args.encoder_type == 'pixel', args.encoder_type
        symbolic = False
    elif args.env_kwargs['observation_mode'] == 'segm':
        assert args.encoder_type == 'segm', args.encoder_type
        symbolic = False
    elif args.env_kwargs['observation_mode'] in PCL_OBS:
        assert args.encoder_type in PCL_MODELS  # symbolic=True
    elif args.env_kwargs['observation_mode'] == 'state':
        assert args.encoder_type == 'mlp'  # symbolic=True
    else:
        raise ValueError(args.env_kwargs['observation_mode'])

    # Mainly for RGB input or visualizing policy rollouts.
    assert args.env_kwargs['camera_width'] == args.env_kwargs['camera_height']
    args.image_size = args.env_kwargs['camera_width']

    # Annoying, mainly because we assume 1000 nvars by default, but sometimes we
    # actually change this (e.g., more for filtering, fewer for debugging).
    n_vars = args.env_kwargs['num_variations']
    args.env_kwargs['cached_states_path'] = (f'{args.env_name}_'
        f'nVars_{str(n_vars).zfill(4)}.pkl')

    # Make wrapper for SoftGym env. Some args can be ignored. Action repeat is
    # confusing here since it's set to be 1, but for MM it ends up being 8...
    env = Env(env=args.env_name,
              symbolic=symbolic,
              seed=args.seed,
              max_episode_length=int(args.env_kwargs['horizon']),
              action_repeat=1,  # ignore this
              bit_depth=8,
              image_dim=args.image_size,
              env_kwargs=args.env_kwargs,
              normalize_observation=False,
              encoder_type=args.encoder_type)
    env.seed(args.seed)
    check_env_version(args, env)
    action_lb, action_ub = get_env_act_bounds(env)

    # Bells and whistles.
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    args.work_dir = logger.get_dir()
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Using different encoder types based on the env and input type.
    IS = args.image_size
    ISC = args.image_size_crop
    if args.encoder_type == 'pixel' and args.env_kwargs['observation_mode'] == 'cam_rgb':
        obs_shape = (3, IS, IS)
        obs_shape_post_aug = (3, ISC, ISC)
    elif args.encoder_type == 'pixel' and args.env_kwargs['observation_mode'] == 'cam_rgbd':
        obs_shape = (4, IS, IS)
        obs_shape_post_aug = (4, ISC, ISC)
    elif args.encoder_type == 'pixel' and args.env_kwargs['observation_mode'] == 'depth_img':
        # Repeats 3 times.
        obs_shape = (3, IS, IS)
        obs_shape_post_aug = (3, ISC, ISC)
    elif args.encoder_type == 'pixel' and args.env_kwargs['observation_mode'] == 'depth_segm':
        # It's (3, IS, IS) for MMOneSphere, (4, IS, IS) for PourWater
        if args.env_name in ['MMOneSphere']:
            kk = 3
        elif args.env_name in ['PourWater', 'PourWater6D']:
            kk = 4
        else:
            raise ValueError(args.env_name)
        obs_shape = (kk, IS, IS)
        obs_shape_post_aug = (kk, ISC, ISC)
    elif args.encoder_type == 'pixel' and args.env_kwargs['observation_mode'] == 'rgb_segm_masks':
        if args.env_name in ['MMOneSphere']:
            kk = 5
        elif args.env_name in ['PourWater', 'PourWater6D']:
            kk = 6
        else:
            raise ValueError(args.env_name)
        obs_shape = (kk, IS, IS)
        obs_shape_post_aug = (kk, ISC, ISC)
    elif args.encoder_type == 'pixel' and args.env_kwargs['observation_mode'] == 'rgbd_segm_masks':
        if args.env_name in ['MMOneSphere']:
            kk = 6
        elif args.env_name in ['PourWater', 'PourWater6D']:
            kk = 7
        else:
            raise ValueError(args.env_name)
        obs_shape = (kk, IS, IS)
        obs_shape_post_aug = (kk, ISC, ISC)
    elif args.encoder_type == 'segm':
        assert hasattr(env, 'n_segm_classes')
        obs_shape = (env.n_segm_classes, IS, IS)
        obs_shape_post_aug = (env.n_segm_classes, ISC, ISC)
    elif args.encoder_type in PCL_MODELS:
        obs_shape = env.observation_space.shape
        obs_shape_post_aug = obs_shape
    elif args.encoder_type == 'mlp':
        obs_shape = env.observation_space.shape
        obs_shape_post_aug = obs_shape
    elif args.encoder_type == 'state_predictor_then_mlp':
        # In this case, the point cloud is the observation. So this is used for BOTH
        # of the data buffers we have to create in this case -- just that one of the
        # buffers has to have the target as the _state_ instead of the action.
        obs_shape = env.observation_space.shape
        obs_shape_post_aug = obs_shape
    else:
        raise ValueError(args.encoder_type)

    # If we use tool flow, the buffer saves tool flow, even though the 'action space'
    # remains whatever the env uses. Also fix for the new rotation representations.
    action_shape = env.action_space.shape
    if args.act_type in FLOW_ACTS:
        obs_pc = env.observation_space.shape
        buffer_action_shape = (obs_pc[0], 3)
    elif args.rotation_representation in NO_RPMG_REPRESENTATIONS:
        if '4D' in args.rotation_representation:
            buffer_action_shape = (3 + 4,)
        elif '6D' in args.rotation_representation:
            buffer_action_shape = (3 + 6,)
        elif '9D' in args.rotation_representation:
            buffer_action_shape = (3 + 9,)
        elif '10D' in args.rotation_representation:
            buffer_action_shape = (3 + 10,)
    elif args.rotation_representation in ROT_REPRESENTATIONS:
        buffer_action_shape = (12,)  # 3D + 9D
    else:
        buffer_action_shape = action_shape
    assert args.act_type in ALL_ACTS

    # Get info shape from env
    info_shape = get_info_shape(env=env, env_name=args.env_name)

    # In this special case, we have to make another buffer with _state_ as the targets.
    # This will be used in phase 1 of the training. Then phase 2 will proceed as normal
    # with point cloud as input to state predictor, _then_ the actions as the target.
    if args.encoder_type == 'state_predictor_then_mlp':
        state_dim = get_state_dim(args.env_name)

        # Be careful about any scaling. I think we should NOT scale any state targs.
        # Save that for the other `data_buffer`.
        data_buffer_state_targs = replay_buffer.PointCloudReplayBuffer(
            args=args,
            obs_shape=obs_shape,
            action_shape=(state_dim,),  # state dim, needs to be in: '(x,)' form
            info_shape=info_shape,
            device=device,
            action_lb=action_lb,
            action_ub=action_ub,
            action_repeat=env._env.action_repeat,
            state_as_target=True,  # NEW
        )

        # Make a state encoder using Classification PointNet++. TODO(daniel)
        print('Making the point cloud to state encoder:')
        pcl_to_state_encoder = PointNet2_Class(
            in_dim=obs_shape[1],
            out_dim=state_dim,
            encoder_type=args.encoder_type,
        ).to(device)
        print(pcl_to_state_encoder)

        # Train the state encoder! Then we'll use it for later.
        train_state_encoder(
            net=pcl_to_state_encoder,
            data=data_buffer_state_targs,
        )
    else:
        pcl_to_state_encoder = None

    # Create buffer from the BC data, noting whether it is filtered or not.
    if args.env_kwargs['observation_mode'] in PCL_OBS:
        ReplayBuffer = replay_buffer.PointCloudReplayBuffer
    elif args.env_kwargs['observation_mode'] == 'state':
        ReplayBuffer = replay_buffer.StateReplayBuffer
    elif args.env_kwargs['observation_mode'] in ['cam_rgb', 'cam_rgbd', 'depth_segm',
            'rgb_segm_masks', 'rgbd_segm_masks', 'depth_img']:
        ReplayBuffer = replay_buffer.ImageReplayBuffer
    else:
        raise ValueError()

    data_buffer = ReplayBuffer(
        args=args,
        obs_shape=obs_shape,
        action_shape=buffer_action_shape,
        info_shape=info_shape,
        device=device,
        action_lb=action_lb,
        action_ub=action_ub,
        action_repeat=env._env.action_repeat,
    )

    # New to BC, handle possibility of filtering by config, as well as
    # handling different #s of training episodes.
    if args.bc_data_filtered:
        configs_train = data_buffer.train_config_idxs
        configs_valid = data_buffer.valid_config_idxs
        env.set_filtered_config_ranges(configs_train, configs_valid)

    # Make BC agent, with policy (i.e., actor), optimizer, etc.
    agent = make_agent(
        obs_shape=obs_shape_post_aug,
        action_shape=action_shape,
        args=args,
        device=device,
    )
    if pcl_to_state_encoder is not None:
        agent.set_state_encoder(pcl_to_state_encoder)

    # Logger: what was used for CURL; chester_logger: from Xingyu / Yufei.
    L = Logger(args.work_dir, use_tb=True, chester_logger=logger)
    n_train_mbs = max(int(data_buffer.num_train_items() / args.batch_size), 1)
    print(f'Batch size: {args.batch_size}. Updates per epoch: {n_train_mbs}')
    t_train = 0.0
    t_eval = 0.0

    # Load tool_reducer if needed
    if args.reduce_tool_points:
        tool_reducer = MixedMediaToolReducer(args, env._env.action_repeat)
    else:
        tool_reducer = None

    # Behavioral Cloning! If just loading, we'll skip training.
    for epoch in range(args.n_epochs+1):
        L.log('train/epoch', epoch, epoch)
        start_t = time.time()
        # Evaluate MSE each epoch, but do rollouts every args.eval_interval epochs.
        with utils.eval_mode(agent):
            evaluate(env, agent, video_dir, L, epoch, args, data_buffer, tool_reducer)
        t_eval += (time.time() - start_t)
        L.log('eval/time_cum', t_eval, epoch)
        if args.load_model:
            print('Exiting due to loading a model.')
            import sys; sys.exit()
        if epoch % args.save_freq == 0:
            agent.save(model_dir, epoch)

        # TODO(daniel) need to update some of these terms.
        t_losses = []
        t_losses_pos = []
        t_losses_rot = []
        t_losses_consistency = []
        t_losses_dense = []
        start_t = time.time()
        for _ in range(n_train_mbs):
            t_loss, t_loss_p, t_loss_q, t_loss_c, t_loss_d = agent.update(data_buffer, epoch=epoch)
            t_losses.append(t_loss)
            t_losses_pos.append(t_loss_p)
            t_losses_rot.append(t_loss_q)
            t_losses_consistency.append(t_loss_c)
            t_losses_dense.append(t_loss_d)
        t_train += (time.time() - start_t)
        L.log('train/MSE_loss', np.mean(t_losses), epoch)  # the combined loss
        L.log('train/MSE_loss_pos', np.mean(t_losses_pos), epoch)
        L.log('train/MSE_loss_rot', np.mean(t_losses_rot), epoch)
        L.log('train/MSE_loss_consistency', np.mean(t_losses_consistency), epoch)
        L.log('train/MSE_loss_dense', np.mean(t_losses_dense), epoch)
        L.log('train/time_cum', t_train, epoch)

        # Log single flow visualization
        if ((agent.act_type in FLOW_ACTS or agent.encoder_type in PCL_COMPRESS and args.log_flow_visuals) and
                (agent.encoder_type not in AVOID_FLOW_PREDS)):
            # Depends on what we want to do. Can use either a random sample (and
            # use first of this) or just use first obs/act info w/relevant info.
            obs, act_gt, _info = data_buffer.sample_obs_act(get_info=True)
            #obs, act_gt, _info = data_buffer.get_first_obs_act_info()
            with torch.no_grad():
                flow_dict = agent.get_single_flow(obs, _info, act_gt)
            xyz = flow_dict['xyz'].detach().cpu().numpy()
            flow = flow_dict['flow'].detach().cpu().numpy()
            gt_flow = flow_dict['act_gt'].detach().cpu().numpy()
            if 'flow_r' not in flow_dict:
                # 3D flow case
                fig = utils.create_flow_gt_plot(xyz, flow, gt_flow, args=args)
            else:
                # 6D flow case
                act = flow_dict['act'].detach().cpu().numpy()
                flow_r = flow_dict['flow_r'].detach().cpu().numpy()
                fig = utils.create_6d_flow_gt_plot(xyz, flow, flow_r, act, gt_flow, args=args)
            logger.logkv('train/flow_gt_plot', fig)

        # Dump here to get data per epoch (might not include some validation stuff).
        L.dump(epoch)

    print('\nDone with BC. Elapsed train / eval time comparison:')
    print(f'  cumulative train time: {t_train:0.2f}s')
    print(f'  cumulative eval time:  {t_eval:0.2f}s')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
