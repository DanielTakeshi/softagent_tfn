import numpy as np
import torch
import os
import time
import json
from gym.spaces import Box

# NOTE(daniel): to reduce error prone code, try and use constants from `bc.bc`.
from bc import utils
from bc import replay_buffer
from bc.bc import (
    BCAgent, PIXEL_ENC, PCL_MODELS, FLOW_ACTS, ALL_ACTS, ENV_ACT_MODES, PCL_COMPRESS
)

# We're unlikely to change the logger or `update_env_kwargs` in BC.
from curl.logger import Logger
from chester import logger
import wandb


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


def evaluate(agent, video_dir, L, epoch, args, data_buffer, logger):
    """Run evaluation step, this is the 'eval' in the logger.

    See documentation in the corresponding CURL / SAC code. We support extra
    features relevant to BC, such as computing validation MSE.
    """
    def run_eval_mse():
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

            # Generating the flow for the entire batch_size in the eval episode.
            eval_flow_dict = agent.get_demo_flow(obs_valid, info_valid, act_valid)

            viz_folder = os.path.join(args.bc_data_dir, 'flow_viz')

            if 'flow_r' not in eval_flow_dict:
                if not os.path.isdir(viz_folder):
                    os.makedirs(viz_folder)
                if vv == 0:
                    for action in range(eval_flow_dict['batch_size']):
                        # Ladle loaction in eval
                        xyz = eval_flow_dict['demo_data'][action]['xyz'].detach().cpu().numpy()
                        # Target location in eval
                        xyz_t = eval_flow_dict['demo_data'][action]['xyz_t'].detach().cpu().numpy()
                        # Tool flow
                        flow = eval_flow_dict['demo_data'][action]['flow'].detach().cpu().numpy()
                        # Tool flow
                        gt_flow = eval_flow_dict['demo_data'][action]['act_gt'].detach().cpu().numpy()
                        eval_fig = utils.create_flow_gt_plot(xyz, xyz_t, flow, gt_flow, args=args)
                        if (epoch%100 == 0):
                            eval_fig.write_html(os.path.join(viz_folder, 'flow_epoch_{}_{}.html'.format(epoch, action)))
                            eval_fig.write_image(os.path.join(viz_folder, 'flp_{}_{}.png'.format(epoch, action)))
            v_loss, v_loss_p, v_loss_q = agent.evaluate_mse(
                    obs=obs_valid, act_gt=act_valid, info=info_valid)
            v_losses.append(v_loss)
            v_losses_pos.append(v_loss_p)
            v_losses_rot.append(v_loss_q)
        logger.logkv('eval/flow_gt_plot', eval_fig)
        agent.plot_eval_preds_and_reset(epoch, video_dir)
        L.log('eval/MSE_loss', np.mean(v_losses), epoch)
        L.log('eval/MSE_loss_pos', np.mean(v_losses_pos), epoch)
        L.log('eval/MSE_loss_rot', np.mean(v_losses_rot), epoch)

    # Check valid MSE to see if it is generalizing to unseen data.
    if not args.test_overfitting:
        run_eval_mse()
    L.dump(epoch)


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


def main(args):
    """Runs Behavioral Cloning.

    Creates `Env` here and if there are no cached configs, generate them. The `Env`
    is a wrapper around the usual normalized sim_env use for quick SoftGym tests. The
    `action_repeat=1` but this is for `Env.step()`, we still use action repeat for
    the internal FlexEnv as specified in env_kwargs.
    """
    if args.seed == -1:
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)

    # Bells and whistles. Later, save flow visuals in `video_dir`?
    assert args.encoder_type in PCL_MODELS, args.encoder_type
    assert args.act_type in ALL_ACTS
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts)
    args.work_dir = logger.get_dir()
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # We don't have a simulation env. Just directly supply env info here.
    # In sim we were using a factor of 1/0.004 as a scaling factor. Actually
    # the naive PN++ needs this scaling to predict translations well.
    sf = args.data_info['scaling_factor']
    if args.data_info['translation_only']:
        action_low  = np.array([-sf, -sf, -sf])
        action_high = np.array([ sf,  sf,  sf])
    else:
        action_low  = np.array([-sf, -sf, -sf, -1, -1, -1])
        action_high = np.array([ sf,  sf,  sf,  1,  1,  1])
    action_space = Box(action_low, action_high, dtype=np.float32)
    max_points = args.data_info['max_points']
    pc_point_dim = args.data_info['n_obs_dim']  # (x,y,z, onehot(tool), onehot(targ), onehot(dist))
    obs_shape = (max_points, pc_point_dim)  # (N,d), note: dist should be all 0
    observation_space = Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
    obs_shape = observation_space.shape
    action_shape = action_space.shape
    obs_dim_keypt = 7   # (position,quaternion) orientation of the tool tip
    info_shape = (obs_dim_keypt,)

    # If we use tool flow, we want the buffer to actually save tool flow.
    if args.act_type in FLOW_ACTS:
        buffer_action_shape = (obs_shape[0], 3)
    else:
        buffer_action_shape = action_shape

    # Make the replay buffer to hold the Behavioral Cloning dataset.
    if args.obs_type == 'point_cloud':
        ReplayBuffer = replay_buffer.PointCloudReplayBuffer
    elif args.obs_type == 'cam_rgb':
        ReplayBuffer = replay_buffer.ImageReplayBuffer
    data_buffer = ReplayBuffer(
        args=args,
        obs_shape=obs_shape,
        action_shape=buffer_action_shape,
        info_shape=info_shape,
        device=device,
        action_lb=action_low,
        action_ub=action_high,
    )

    # Make BC agent, with policy (i.e., actor), optimizer, etc.
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device,
    )

    # Logger: what was used for CURL; chester_logger: from Xingyu / Yufei.
    L = Logger(args.work_dir, use_tb=True, chester_logger=logger)
    n_train_mbs = int(data_buffer.num_train_items() / args.batch_size)
    print(f'Batch size: {args.batch_size}. Updates per epoch: {n_train_mbs}')
    t_train = 0.0
    t_eval = 0.0

    # Behavioral Cloning! If just loading, we'll skip training.
    for epoch in range(args.n_epochs + 1):
        L.log('train/epoch', epoch, epoch)
        start_t = time.time()
        with utils.eval_mode(agent):
            evaluate(agent, video_dir, L, epoch, args, data_buffer, logger)
        t_eval += (time.time() - start_t)
        L.log('eval/time_cum', t_eval, epoch)
        if epoch % args.save_freq == 0:
            agent.save(model_dir, epoch)

        t_losses = []
        t_losses_pos = []
        t_losses_rot = []
        t_losses_consistency = []
        start_t = time.time()
        for _ in range(n_train_mbs):
            t_loss, t_loss_p, t_loss_q, t_loss_c = agent.update(data_buffer)
            t_losses.append(t_loss)
            t_losses_pos.append(t_loss_p)
            t_losses_rot.append(t_loss_q)
            t_losses_consistency.append(t_loss_c)
        t_train += (time.time() - start_t)
        L.log('train/MSE_loss', np.mean(t_losses), epoch)
        L.log('train/MSE_loss_pos', np.mean(t_losses_pos), epoch)
        L.log('train/MSE_loss_rot', np.mean(t_losses_rot), epoch)
        L.log('train/MSE_loss_consistency', np.mean(t_losses_consistency), epoch)
        L.log('train/time_cum', t_train, epoch)

        # Log single flow visualization
        if (agent.act_type in FLOW_ACTS or agent.encoder_type in PCL_COMPRESS and
                args.log_flow_visuals):
            # Depends on what we want to do. Can use either a random sample (and
            # use first of this) or just use first obs/act info w/relevant info.
            obs, act_gt, _info = data_buffer.sample_obs_act(get_info=True)
            #obs, act_gt, _info = data_buffer.get_first_obs_act_info()
            with torch.no_grad():
                flow_dict = agent.get_single_flow(obs, _info, act_gt)
            xyz = flow_dict['xyz'].detach().cpu().numpy()
            xyz_t = flow_dict['xyz_t'].detach().cpu().numpy()
            flow = flow_dict['flow'].detach().cpu().numpy()
            gt_flow = flow_dict['act_gt'].detach().cpu().numpy()

            if 'flow_r' not in flow_dict:
                # 3D flow case
                fig = utils.create_flow_gt_plot(xyz, xyz_t, flow, gt_flow, args=args)
            else:
                # 6D flow case
                act = flow_dict['act'].detach().cpu().numpy()
                flow_r = flow_dict['flow_r'].detach().cpu().numpy()
                fig = utils.create_6d_flow_gt_plot(xyz, flow, flow_r, act, gt_flow, args=args)
            logger.logkv('train/flow_gt_plot', fig)

    print('\nDone with BC. Elapsed train / eval time comparison:')
    print(f'  cumulative train time: {t_train:0.2f}s')
    print(f'  cumulative eval time:  {t_eval:0.2f}s')


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
