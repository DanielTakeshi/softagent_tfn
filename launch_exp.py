"""Launch experiments.

If running `--debug` we only run the first of the VariantGenerators. Otherwise,
we'll run all the different combinations (typically with a multiple of random
seeds), but also in parallel on the same GPU, so watch out about RAM.
"""
import os
from os.path import join
import time
import torch
import click
import socket
from chester.run_exp import run_experiment_lite, VariantGenerator
from softgym.registered_env import env_arg_dict
from bc.train import run_task
from bc.bc import VALID_COMBOS, PCL_MODELS, PCL_OBS
from bc import exp_configs

# ----------------------------- ADJUST -------------------------------- #
# BC data directory. This is an example, you need to adjust.
DATA_HEAD = '/data/seita/softgym_mm/data_demo'

# Saved models, for rolling out policies later.
LOAD_HEAD = '/data/seita/softagent_mm'
# --------------------------------------------------------------------- #


def get_bc_data_dir(env, env_version, alg_policy, filtered, mode, nvars,
        act_mode=None):
    """Get BC data directory.

    This is machine-dependent and user-dependent.  Assumes we already have data
    stored somewhere, and this will help us find the correct data path to use.

    Parameters
    ----------
    alg_policy: string indicating algorithmic policy we used for generating
        the behavioral cloning data.
    filtered: True if we filter by success, which is why we have a higher nvars
        so that we can get to a target number of successes.
    mode: either seuss or local, in which case I have to change directories.
    nvars: integer indicating the number of cached configs that we used for
        generating the data. This is higher than the number of actual training
        data configs (episodes) we will use.
    act_mode: string indicating the action mode.
    """
    filt = 'filtered' if filtered else 'UNfiltered'
    suffix = (f'{env}_{env_version}_BClone_{filt}_wDepth_{alg_policy}_nVars_{nvars}'
            f'_obs_combo')
    if act_mode is not None:
        suffix = f'{suffix}_act_{act_mode}'
    if env in ['PourWater', 'PourWater6D']:
        suffix += '_withWaterFrac'
    bc_data_dir = join(DATA_HEAD, suffix)
    return bc_data_dir


def get_exp_prefix(env, env_version, n_train_demos, obs_type, encoder_type,
        act_type, action_repeat, horizon, DoFs, scale_pcl_flow=False,
        scale_targets=False, remove_zeros_PCL=False, load_model=False,
        remove_skip_connections=False, gaussian_noise_PCL=0.0, reduce_tool_PCL='None',
        data_augm_PCL='None', zero_center_PCL=False, rotation_representation='None',
    ):
    """We save directories using this prefix."""
    if act_type == 'eepose_convert':
        act_type = f'eepose_convert_{rotation_representation}'
    exp_prefix = (f'BC04_{env}_{env_version}_ntrain_'
            f'{str(n_train_demos).zfill(4)}_{obs_type}_{encoder_type}_'
            f'{act_type}_{DoFs}DoF_ar_{action_repeat}_hor_{horizon}')
    if remove_skip_connections:
        exp_prefix += f'_NOSKIP'

    if encoder_type in PCL_MODELS:
        # Assume by default: no tool reducing, remove 0s, gauss noise 0.
        if reduce_tool_PCL != 'None':
            exp_prefix += f'_reducetool'
        if gaussian_noise_PCL > 0:
            exp_prefix += f'_GaussNoise_{gaussian_noise_PCL}'
        if data_augm_PCL != 'None':
            exp_prefix += f'_dataAug_{data_augm_PCL}'
        if not remove_zeros_PCL:
            exp_prefix += f'_keep_0s'
        if scale_pcl_flow:
            exp_prefix += f'_scalePCL'
            if zero_center_PCL:
                exp_prefix = exp_prefix.replace('_scalePCL','_scaleZeroPCL')
        else:
            exp_prefix += f'_rawPCL'
            if zero_center_PCL:
                exp_prefix = exp_prefix.replace('_rawPCL','_rawZeroPCL')

    if scale_targets:
        exp_prefix += f'_scaleTarg'
    else:
        exp_prefix += f'_noScaleTarg'

    # Simplify some stuff from file name to avoid excessively long directories.
    if 'point_cloud_' in exp_prefix:
        exp_prefix = exp_prefix.replace('point_cloud_', 'PCL_')
    if 'pointnet_' in exp_prefix:
        exp_prefix = exp_prefix.replace('pointnet_', 'PNet2_')

    if load_model:
        exp_prefix = f'{exp_prefix}_load_model'
    return exp_prefix


def get_saved_model_path(exp_prefix, load_epoch, load_index=0):
    """For finding model paths for loading BC-trained polices.

    This assumes we will use the exp_prefix indicated here to identify
    the desired model directory. Also there are usually a bunch of these
    so I set the model index as a function of the sorted list.
    """
    saved_model_dir = join(LOAD_HEAD, exp_prefix.replace('_load_model',''))
    saved_model_trials = sorted(
        [join(saved_model_dir,x) for x in os.listdir(saved_model_dir)])
    saved_model_trial = saved_model_trials[load_index]
    saved_models = join(saved_model_trial, 'model')
    models = sorted([join(saved_models,x) for x in os.listdir(saved_models)])
    if load_epoch == -1:
        model_path = models[-1]
    else:
        model_path = join(saved_models, f'ckpt_{str(load_epoch).zfill(4)}.tar')
        assert model_path in models, model_path
    assert os.path.exists(model_path), model_path
    print(f'Behavioral Cloning, LOADING model:\n{model_path}\n')
    return model_path


def get_act_info_from_alg(env, alg_policy):
    """Given algorithmic policy type, we should return the action mode.

    As a reminder, this overrides defaults in `softgym/registered_env.py`.
    To be clear: each alg_policy was implemented using a particular action
    mode, so that is the format we will want the leaned agent to use when
    it predicts actions.

    Returns: (dofs, act_mode, act_repeat, horizon).
        dofs: DoFs used by demonstrator (may be deliberately restrictive)
        act_mode: the SoftGym env's 'action_mode', e.g. axis_angle
        act_repeat: the SoftGym env's 'action_repeat'
        horizon: the SoftGym env's 'horizon'.
    The # of steps we take is act_repeat times horizon. At test time for
    BC, we can increase the horizon if needed to go to 800.
    """
    if alg_policy == 'ladle_algorithmic_v01':
        # Just for Spheres envs, used for debugging multi-sphere stuff.
        return (3, 'translation', 8, 100)
    elif alg_policy == 'ladle_algorithmic_v02':
        return (3, 'translation', 8, 100)
    elif alg_policy == 'ladle_algorithmic_v04':
        return (4, 'translation_axis_angle', 8, 100)
    elif alg_policy == 'ladle_algorithmic_v05':
        return (3, 'translation', 1, 600)
    elif alg_policy == 'ladle_algorithmic_v06':
        return (4, 'translation_axis_angle', 1, 600)
    elif alg_policy == 'ladle_algorithmic_v07':
        return (4, 'translation_axis_angle', 1, 600)
    elif alg_policy == 'ladle_algorithmic_v08':
        return (3, 'translation', 8, 100)
    elif alg_policy == 'ladle_6dof_rotations_scoop':
        return (6, 'translation_axis_angle', 8, 100)
    elif alg_policy == 'pw_algo_v02':
        if env == 'PourWater':
            return (3, 'translation_axis_angle', 8, 100)
        elif env == 'PourWater6D':
            return (6, 'translation_axis_angle', 8, 100)
    else:
        raise ValueError(alg_policy)


def do_we_project_axis_ang_y(alg_policy, act_mode):
    """Feels hacky, projects an axis angle (wx,wy,wz) to (0,wy,0).

    This is mainly if our demonstrator deliberately restricted itself to
    a particular DoF. Doesn't apply for newer 6DoF scooping demos (thankfully).
    """
    project_axis_ang_y = False
    if ((alg_policy in ['ladle_algorithmic_v04',
                        'ladle_algorithmic_v06',
                        'ladle_algorithmic_v07'])
        and (act_mode == 'translation_axis_angle')
    ):
        project_axis_ang_y = True
    return project_axis_ang_y


def do_we_project_pour_water(alg_policy, act_mode, env):
    """Feels hacky, projects (x,y,z,wx,wy,wz) to (x,y,0,0,0,wz)."""
    project_pour_water = False
    if ((alg_policy in ['pw_algo_v02'])
        and (act_mode == 'translation_axis_angle')
        and (env == 'PourWater')
    ):
        project_pour_water = True
    return project_pour_water


def do_we_weight_R_t_loss(alg_policy, act_mode, encoder_type, scale_targets):
    """Weigh the losses for translation and rotation separately.

    If we do not do this (default) then we have equal weight on both. This
    is only relevant for methods that will directly output some rotation
    value directly. Be careful when scaling the targets; typically we scale
    the translation but not the rotation, so that generally means increasing
    the weight of the rotation part for loss function purposes.
    """
    weighted_R_t_loss = False
    if ((alg_policy in ['pw_algo_v02',
                        'ladle_algorithmic_v04',
                        'ladle_algorithmic_v06',
                        'ladle_algorithmic_v07',
                        'ladle_6dof_rotations_scoop'])
        and (act_mode == 'translation_axis_angle')
        and (encoder_type != 'pointnet_svd_pointwise')
    ):
        weighted_R_t_loss = True
    weighted_R_t_loss = (weighted_R_t_loss and scale_targets)
    return weighted_R_t_loss


def get_nvars(env, env_version):
    """What we used for the number of cached configs.

    This must match, otherwise we have to reload cached configs at the start, and
    that could add another factor of variation.
    """
    if env == 'PourWater':
        nvars = 1500
    elif env == 'MMOneSphere' and env_version == 'v02':
        nvars = 1200
    else:
        nvars = 2000
    return nvars


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=False)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    # Only do this if loading a model (and NOT training). By default load the
    # last snapshot corresponding to the current `vg` settings with seed 100.
    # Also load data buffer so we can check predictions on same things.
    load_model = False
    load_epoch = -1

    # If using, this needs to be the FULL absolute path, including the `ckpt_XYZ.tar`.
    load_model_path = ''

    # ------------------------------ ADJUST ---------------------------------- #
    # (1) PourWater, 3DoF (2 transl, 1 rot)
    #env, env_version, alg_policy = 'PourWater', 'v01', 'pw_algo_v02'

    # (2) PourWater, 6DoF :)
    env, env_version, alg_policy = 'PourWater6D', 'v01', 'pw_algo_v02'

    # We ran these for MMOneSphere, but can't be run without closed source code.
    #env, env_version, alg_policy = 'MMOneSphere', 'v01', 'ladle_algorithmic_v02'
    #env, env_version, alg_policy = 'MMOneSphere', 'v01', 'ladle_algorithmic_v04'
    #env, env_version, alg_policy = 'MMOneSphere', 'v02', 'ladle_6dof_rotations_scoop'

    # This is a debugging env, SpheresLadle, 3DoF. v03 = 1 sphere, v02 = 2 spheres.
    #env, env_version, alg_policy = 'SpheresLadle', 'v03', 'ladle_algorithmic_v01'
    #env, env_version, alg_policy = 'SpheresLadle', 'v02', 'ladle_algorithmic_v01'
    # ------------------------------------------------------------------------ #

    # Bells and whistles. Filtered means we only consider successful examples.
    filtered = True
    nvars = get_nvars(env, env_version)
    DoFs, act_mode, action_repeat, horizon = get_act_info_from_alg(env, alg_policy)
    bc_data_dir = get_bc_data_dir(env, env_version, alg_policy, filtered,
            mode=mode, nvars=nvars, act_mode=act_mode)

    # Default of 100 train demos, except for other special cases.
    n_train_demos = 100
    if env == 'SpheresLadle':
        n_train_demos = 1000
    elif env == 'MMOneSphere' and env_version == 'v02':
        n_train_demos = 25

    # ------------------------------ ADJUST ---------------------------------- #
    # Pick the method to use. This is ToolFlowNet:
    #   TFN (3D flow) w/SVD + consist.:  exp_configs.SVD_POINTWISE_EE2FLOW
    #
    # This is an alternative which predicts '6D flow', or predicts rotation and
    # translation separately. This got similar performance as ToolFlowNet.
    #   TFN (6D flow) w/SVD + consist.:  exp_configs.SVD_POINTWISE_6D_EE2FLOW_SVD
    #
    # Baselines:
    #   Direct Vector MSE: exp_configs.NAIVE_CLASS_PN2_TO_VECTOR_6DoF (old way)
    #   Direct Vector MSE: exp_configs.DIRECT_VECTOR_INTRINSIC_AXIS_ANGLE (new way)
    #   Direct Vector PW.: exp_configs.NAIVE_CLASS_PN2_TO_VECTOR_POINTWISE
    #   Dense Transf. MSE: exp_configs.DENSE_TRANSF_POLICY_TIP_6DoF_MSE (old way)
    #   Dense Transf. MSE: exp_configs.DENSE_TRANSF_INTRINSIC_AXIS_ANGLE_6DoF_MSE (new way)
    #   Image CNN:         exp_configs.NAIVE_CNN_6DoF
    #   Image CNN (RBGD):  exp_configs.NAIVE_CNN_RGBD_6DoF
    #   Image CNN (DSEGM): exp_configs.NAIVE_CNN_DEPTH_SEGM_6DoF
    #   TFN MSE after SVD: exp_configs.SVD_3D_FLOW_EEPOSE_MSE_LOSS
    #   TFN PM before SVD: exp_configs.SVD_PRE_POINTWISE_3D_EE2FLOW_SVD
    #   State baseline:    exp_configs.STATE_POLICY_BASELINE
    #
    # Post-CoRL acceptance, test with other rotation representations:
    #   6D rotations:      exp_configs.DIRECT_VECTOR_6D_ROTATION
    # ------------------------------------------------------------------------ #
    this_cfg = exp_configs.SVD_POINTWISE_EE2FLOW  # ToolFlowNet
    #this_cfg = exp_configs.DIRECT_VECTOR_INTRINSIC_AXIS_ANGLE  # Direct Vector MSE
    # ------------------------------------------------------------------------ #

    # Update config. For losses, 'default lambda' is 1 for pointwise / MSE.
    cfg = exp_configs.DEFAULT_CFG
    cfg.update(**this_cfg)
    obs_type = cfg['obs_type']
    act_type = cfg['act_type']
    rotation_representation = cfg['rotation_representation']
    encoder_type = cfg['encoder_type']
    remove_zeros_PCL = cfg['remove_zeros_PCL']
    zero_center_PCL = cfg['zero_center_PCL']
    method_flow2act = cfg['method_flow2act']
    use_dense_loss = cfg['use_dense_loss']
    use_consistency_loss = cfg['use_consistency_loss']
    lambda_dense = cfg['lambda_dense']
    lambda_consistency = cfg['lambda_consistency']
    lambda_pos = cfg['lambda_pos']
    lambda_rot = cfg['lambda_rot']
    rpmg_lambda = cfg['rpmg_lambda']
    scale_pcl_flow = cfg['scale_pcl_flow']
    scale_pcl_val = cfg['scale_pcl_val']
    scale_targets = cfg['scale_targets']
    use_geodesic_dist = cfg['use_geodesic_dist']
    data_augm_img = cfg['data_augm_img']
    data_augm_PCL = cfg['data_augm_PCL']
    image_size_crop = cfg['image_size_crop']
    separate_MLPs_R_t = cfg['separate_MLPs_R_t']
    dense_transform = cfg['dense_transform']
    remove_skip_connections = cfg['remove_skip_connections']
    gaussian_noise_PCL = cfg['gaussian_noise_PCL']
    reduce_tool_PCL = cfg['reduce_tool_PCL']
    reduce_tool_points = cfg['reduce_tool_points']
    tool_point_num = cfg['tool_point_num']
    log_flow_visuals = cfg['log_flow_visuals']

    # Hard-coded BC parameters used for the CoRL 2022 paper.
    n_epochs = 500  # overall passes through data
    eval_interval = 25  # evaluate every 25 snapshots
    n_eval_episodes = 25  # just the first 25
    save_freq = 25  # save snapshots
    n_valid_demos = 25  # just using 25
    test_each_valid_once = True  # we want to touch once each eval

    # Special handling for the MM or spheres envs.
    if env == 'MMMultiSphere':
        assert n_epochs >= 1000, f'We probably want more epochs ({n_epochs})'
    if env != 'SpheresLadle':
        assert eval_interval == n_eval_episodes == 25
        assert n_valid_demos == save_freq == 25

    # Special handling for the MM or spheres envs.
    if env in ['MMOneSphere', 'MMMultiSphere', 'SpheresLadle']:
        if obs_type == 'point_cloud_gt_v01':
            bc_data_dir = bc_data_dir.replace('obs_combo_', 'obs_combo_gt_v01_')
        elif obs_type == 'point_cloud_gt_v02':
            bc_data_dir = bc_data_dir.replace('obs_combo_', 'obs_combo_gt_v02_')
        print(f'With obs {obs_type}, we want {bc_data_dir}')

        if act_type == 'translation' and encoder_type == 'pointnet_svd_pointwise':
            assert act_type == 'flow', f'Do not use ee2flow: {act_type}'

    # Some additional random checks / stuff to reduce error-prone code.
    if scale_pcl_flow:
        assert not scale_targets, 'Do not do this with scaling PCL'
    if obs_type in PCL_OBS:
        assert (act_mode, act_type, encoder_type) in VALID_COMBOS, \
            f'bad combo: {act_mode} {act_type} {encoder_type}'
    if dense_transform:
        assert obs_type in PCL_OBS
    batch_size = 128
    if obs_type in PCL_OBS:
        batch_size = 16
    if remove_skip_connections:
        assert encoder_type in ['pointnet_svd_pointwise',
                'pointnet_svd_pointwise_6d_flow']
    if use_consistency_loss:
        if env == 'PourWater':
            assert lambda_consistency == 0.5, lambda_consistency
        elif env in ['MMOneSphere', 'MMMultiSphere']:
            assert lambda_consistency == 0.1, lambda_consistency
        elif env in ['SpheresLadle']:
            assert lambda_consistency == 0.1, lambda_consistency

    # Automatically get exp prefix from settings.
    exp_prefix = get_exp_prefix(
            env=env, env_version=env_version, n_train_demos=n_train_demos,
            obs_type=obs_type, encoder_type=encoder_type, act_type=act_type,
            action_repeat=action_repeat, horizon=horizon, DoFs=DoFs,
            scale_pcl_flow=scale_pcl_flow, scale_targets=scale_targets,
            remove_zeros_PCL=remove_zeros_PCL, load_model=load_model,
            remove_skip_connections=remove_skip_connections,
            gaussian_noise_PCL=gaussian_noise_PCL, reduce_tool_PCL=reduce_tool_PCL,
            data_augm_PCL=data_augm_PCL, zero_center_PCL=zero_center_PCL,
            rotation_representation=rotation_representation,
    )

    # Again, only relevant if loading. Save models if training.
    if load_model and load_model_path == '':
        load_model_path = get_saved_model_path(exp_prefix, load_epoch)
    save_model = False if load_model else True

    # These feel like hacks.
    project_axis_ang_y = do_we_project_axis_ang_y(alg_policy, act_mode)
    project_pour_water = do_we_project_pour_water(alg_policy, act_mode, env)
    assert not (project_axis_ang_y and project_pour_water)
    weighted_R_t_loss = do_we_weight_R_t_loss(alg_policy, act_mode, encoder_type,
        scale_targets)
    if weighted_R_t_loss:
        # This was something we originally tested to weigh R and t losses, but
        # we did not see a benefit from this.
        pass

    # ------------------------------------------------------------------------------ #
    # NOTE(daniel): `env_kwargs` starts from SoftGym's `registered_env.py` args; if
    # we assign args with prefix `env_kwargs_` then those have precedence.
    # Be mindful of CPU/GPU RAM limits, but we can make the buffer capacity large.
    # ------------------------------------------------------------------------------ #
    vg = VariantGenerator()
    vg.add('wandb_project', [''])  # Fill this in!
    vg.add('wandb_entity', [''])  # Fill this in!
    vg.add('env_name', [env])
    vg.add('env_kwargs', lambda env_name: [env_arg_dict[env_name]])
    vg.add('env_kwargs_observation_mode', [obs_type])
    vg.add('env_kwargs_action_mode', [act_mode])  # e.g., `translation`; overrides env defaults
    vg.add('env_kwargs_num_variations', [nvars])
    vg.add('env_kwargs_deterministic', [False])
    vg.add('env_kwargs_camera_width', [128])
    vg.add('env_kwargs_camera_height', [128])
    vg.add('env_kwargs_action_repeat', [action_repeat])
    vg.add('env_kwargs_horizon', [horizon])
    vg.add('env_version', [env_version])
    vg.add('image_size_crop', [image_size_crop])
    vg.add('agent', ['bc'])
    vg.add('algorithm', ['BC'])
    vg.add('alg_policy', [alg_policy])
    vg.add('act_type', [act_type])  # the BC _targets_, not to be confused with `act_mode`
    vg.add('rotation_representation', [rotation_representation])
    vg.add('bc_data_dir', [bc_data_dir])
    vg.add('bc_data_filtered', [filtered])
    vg.add('encoder_type', [encoder_type])
    vg.add('method_flow2act', [method_flow2act])  # action selection, flow -> (action_type)
    vg.add('hidden_dim', [256])  # for CNNs only
    vg.add('num_layers', [4])  # for CNNs only
    vg.add('num_filters', [32])  # for CNNs only
    vg.add('actor_lr', [1e-4])
    vg.add('n_epochs', [n_epochs])
    vg.add('n_train_demos', [n_train_demos])  # new, will subsample if needed
    vg.add('n_valid_demos', [n_valid_demos])  # total amount of validation samples
    vg.add('n_eval_episodes', [n_eval_episodes])  # per evaluation stage!
    vg.add('test_each_valid_once', [test_each_valid_once])
    vg.add('data_buffer_capacity', [200000]) # we never get close to 200K for BC
    vg.add('batch_size', [batch_size])
    vg.add('save_video', [True])
    vg.add('save_eval_PCLs', [False])
    vg.add('save_model', [save_model])
    vg.add('load_model', [load_model])  # do we load from prior training run
    vg.add('load_model_path', [load_model_path])  # and what file name?
    vg.add('save_freq', [save_freq])
    vg.add('log_interval', [1])  # not being used now
    vg.add('eval_interval', [eval_interval])
    vg.add('project_axis_ang_y', [project_axis_ang_y])
    vg.add('project_pour_water', [project_pour_water])
    vg.add('test_overfitting', [False])  # should normally be False
    vg.add('log_flow_visuals', [log_flow_visuals])  # was true by default earlier
    vg.add('scale_pcl_flow', [scale_pcl_flow])
    vg.add('scale_pcl_val', [scale_pcl_val])
    vg.add('scale_targets', [scale_targets])  # careful!
    vg.add('weighted_R_t_loss', [weighted_R_t_loss])
    vg.add('lambda_pos', [lambda_pos])
    vg.add('lambda_rot', [lambda_rot])
    vg.add('rpmg_lambda', [rpmg_lambda])
    vg.add('use_consistency_loss', [use_consistency_loss])
    vg.add('lambda_consistency', [lambda_consistency])
    vg.add('use_dense_loss', [use_dense_loss])
    vg.add('lambda_dense', [lambda_dense])
    vg.add('use_geodesic_dist', [use_geodesic_dist])
    vg.add('data_augm_img', [data_augm_img])
    vg.add('data_augm_PCL', [data_augm_PCL])
    vg.add('remove_zeros_PCL', [remove_zeros_PCL])
    vg.add('zero_center_PCL', [zero_center_PCL])
    vg.add('separate_MLPs_R_t', [separate_MLPs_R_t])
    vg.add('dense_transform', [dense_transform])
    vg.add('remove_skip_connections', [remove_skip_connections])
    vg.add('gaussian_noise_PCL', [gaussian_noise_PCL])  # new for PCL obs
    vg.add('reduce_tool_PCL', [reduce_tool_PCL])  # for reducing tool points test
    vg.add('reduce_tool_points', [reduce_tool_points])  # ONLY FOR SCOOPING
    vg.add('tool_point_num', [tool_point_num])
    vg.add('seed', [100])

    if not debug:
        pass
    else:
        pass
        exp_prefix += '_debug'

    print('Number of experiment configurations: ', len(vg.variants()))
    print("exp_prefix: ", exp_prefix)

    hostname = socket.gethostname()
    gpu_num = torch.cuda.device_count()

    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 10:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        if mode in ['seuss', 'autobot']:
            if idx == 0:
                # NOTE(daniel): if we don't need to compile, set to None!
                # For the first experiment, compile the current softgym
                compile_script = None #'./compile_1.0.sh'
                wait_compile = None
            else:
                # Wait 300 seconds for compilation to finish
                compile_script = None
                wait_compile = 300
        elif mode == 'ec2':
            compile_script = 'compile_1.0.sh'
            wait_compile = None
        else:
            compile_script = wait_compile = None
        if hostname.startswith('autobot') and gpu_num > 0:
            env_var = {'CUDA_VISIBLE_DEVICES': str(idx % gpu_num)}
        else:
            env_var = None
        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            env=env_var
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
