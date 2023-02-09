"""Launch BC experiments for MM.

Based on the CURL/SAC script that we have been using:
    python experiments/curl/launch_curl_mm.py

If running `--debug` we only run the first of the VariantGenerators. Otherwise,
we'll run all the different combinations (typically with a multiple of random
seeds), but also in parallel on the same GPU, so watch out about RAM.

For a description of different datasets, please see `bc/README.md`.
"""
from collections import defaultdict
from os.path import join
import time
import torch
import click
import socket
import numpy as np
from chester.run_exp import run_experiment_lite, VariantGenerator
from bc.train import run_task
from bc.bc import VALID_COMBOS, PCL_MODELS
from bc import exp_configs

# For BC data directory. This may be machine-dependent.
#* Pointers to where the data tarballs are stored. First one for the cluster and the second for marvin/takeshi
# ----------------------------- ADJUST -------------------------------- #
DATA_HEAD = '/data/sarthak/softagent_tfn_physical/data_demo/'
# --------------------------------------------------------------------- #

def get_dataset_info(suffix, mode, num_samples):
    """Given a dataset, return dict with relevant info.

    This should prevent a lot of error-prone hard-coded variables. When
    adding a new data, put the information here. See also:
    https://www.notion.so/Tool-Flow-Experiments-Summary-of-Datasets-7f4f8b3c82a544d1bf7cd5fb1b474c46
    for further info on the datasets.
    """
    info = {}

    info['bc_data_dir'] = join(DATA_HEAD, suffix)
    info['suffix'] = suffix

    # Assume not translation by default unless specified otherwise.
    info['translation_only'] = False

    # WARNING! This has caused a lot of confusion!  TL;DR will divide by this
    # (scales by 250X) for stabilizing training when adding to data replay
    # buffer, then we must scale back down (see `inference.py`). BUT, careful
    # if using flow because the pointwise loss needs to have aligned units.
    info['scaling_factor'] = 0.004

    # This parameter describes the number of points in the pointclouds. Some of the human datasets have 1200
    # points, with 1100 tool points and 100 target points. This incldues 
    # the data used to train the CoRL models. The newer algorithmic dataset has 1400 points, with 1100 tool points
    # and 300 target points.

    info['max_points'] = 1400

    # This is the observer dimension, that is set to 5 as default in case we are testing out "regular"
    # datasets that don't have any distractors. For the "simple" datasets, there are just 4 dimensions.
    info['n_obs_dim'] = 5

    # Assign properties based on the dataset.
    if suffix == 'v01_physicalRotations_pkls':
        info['n_train_demos'] = 39
        info['n_valid_demos'] = 5
    elif suffix == 'v02_physicalTranslations_pkls':
        info['n_train_demos'] = 30
        info['n_valid_demos'] = 6
        info['translation_only'] = True
    elif suffix == 'v03_physicalTranslations_pkls':
        info['n_train_demos'] = 31
        info['n_valid_demos'] = 5
        info['translation_only'] = True
    elif suffix == 'v04_physicalTranslations_pkls':
        info['n_train_demos'] = 24
        info['n_valid_demos'] = 5
        info['translation_only'] = True
    elif suffix == 'v03_physicalTranslations_pkls_k_step_4':
        info['n_train_demos'] = 31
        info['n_valid_demos'] = 5
        info['translation_only'] = True
    elif suffix == 'v03_physicalTranslations_pkls_combined_k_step_4_backup_n_by_5':
        info['n_train_demos'] = 123
        info['n_valid_demos'] = 25
        info['translation_only'] = True
        info['max_points'] = 1200
    elif suffix == 'v03_physicalTranslations_pkls_combined_k_step_4_denser_n_by_5':
        info['n_train_demos'] = 133
        info['n_valid_demos'] = 25
        info['translation_only'] = True
        info['max_points'] = 1400
    elif suffix == 'v02_physicalRotations_pkls':
        info['n_train_demos'] = 29
        info['n_valid_demos'] = 5
    elif suffix == 'v01_02_physicalRotations_pkls':
        info['n_train_demos'] = 68
        info['n_valid_demos'] = 10
    elif suffix == 'v04_demonstrator_translation':
        info['n_train_demos'] = 100
        info['n_valid_demos'] = 25
        info['translation_only'] = True
    elif suffix == 'v04_demonstrator_translation_k_step_4_n_by_5':
        info['n_train_demos'] = 100
        info['n_valid_demos'] = 25
        info['translation_only'] = True
        info['n_obs_dim'] = 5
        info['max_points'] = 1400
    elif suffix == 'v04_demonstrator_translation_n_by_5':
        info['n_train_demos'] = 100
        info['n_valid_demos'] = 25
        info['translation_only'] = True
        info['n_obs_dim'] = 5
        info['max_points'] = 1400
    elif suffix == 'v05_fast_demonstrator':
        info['n_train_demos'] = 100
        info['n_valid_demos'] = 25
        info['translation_only'] = True
        info['n_obs_dim'] = 5
        info['scaling_factor'] = 0.02
        info['max_points'] = 1400
    elif suffix == 'v05_fast_demonstrator_variable_composing':
        info['n_train_demos'] = 100
        info['n_valid_demos'] = 25
        info['translation_only'] = True
        info['n_obs_dim'] = 5
        info['scaling_factor'] = 0.02
        info['max_points'] = 1400
    elif suffix == 'v06_human_fast_zero_lag_k_steps_2':
        info['n_train_demos'] = 100
        info['n_valid_demos'] = 25
        info['translation_only'] = True
        info['n_obs_dim'] = 5
        info['scaling_factor'] = 0.01
        info['max_points'] = 1400
    elif suffix == 'v06_human_fast_zero_lag_variable_composing':
        info['n_train_demos'] = 100
        info['n_valid_demos'] = 25
        info['translation_only'] = True
        info['n_obs_dim'] = 5
        info['scaling_factor'] = 0.01
        info['max_points'] = 1400
    elif suffix == 'v07_rotation_translation_variably_composed':
        info['n_train_demos'] = 105
        info['n_valid_demos'] = 25
        info['n_obs_dim'] = 5
        info['scaling_factor'] = 0.05
        info['max_points'] = 1400
    elif suffix == 'v01_simple_algorithmic_dataset_x_move_n_by_4_stop':
        info['n_train_demos'] = 45
        info['n_valid_demos'] = 5
        info['translation_only'] = True
    elif suffix == 'v01_simple_algorithmic_dataset_x_move_n_by_6':
        info['n_train_demos'] = 45
        info['n_valid_demos'] = 5
        info['translation_only'] = True
    elif suffix == 'v01_simple_algorithmic_dataset_y_move_n_by_4':
        info['n_train_demos'] = 46
        info['n_valid_demos'] = 5
        info['translation_only'] = True
    elif suffix == 'v01_simple_algorithmic_dataset_y_move_n_by_6':
        info['n_train_demos'] = 46
        info['n_valid_demos'] = 5
        info['translation_only'] = True
    elif suffix == 'v04_demonstrator_translation' and num_samples == '10':
        info['n_train_demos'] = 8
        info['n_valid_demos'] = 2
        info['translation_only'] = True
    elif suffix == 'v04_demonstrator_translation' and num_samples == '30':
        info['n_train_demos'] = 25
        info['n_valid_demos'] = 5
        info['translation_only'] = True
    elif suffix == 'v04_demonstrator_translation' and num_samples == '50':
        info['n_train_demos'] = 40
        info['n_valid_demos'] = 10
        info['translation_only'] = True
    elif suffix == 'v04_demonstrator_translation' and num_samples == '80':
        info['n_train_demos'] = 70
        info['n_valid_demos'] = 10
        info['translation_only'] = True
    elif suffix == 'v04_demonstrator_translation' and num_samples == '100':
        info['n_train_demos'] = 80
        info['n_valid_demos'] = 20
        info['translation_only'] = True
    else:
        raise ValueError(suffix)

    return info


def get_exp_prefix(n_train_demos, suffix, obs_type, encoder_type,
        act_type, scale_pcl_flow=False, scale_targets=False):
    """We save directories using this prefix."""
    exp_prefix = (f'BCphy_{suffix}_ntrain_'
            f'{str(n_train_demos).zfill(4)}_{obs_type}_{encoder_type}_'
            f'acttype_{act_type}')
    if encoder_type in PCL_MODELS:
        if scale_pcl_flow:
            exp_prefix += f'_scalePCL'
        else:
            exp_prefix += f'_rawPCL'
    if scale_targets:
        exp_prefix += f'_scaleTarg'
    else:
        exp_prefix += f'_noScaleTarg'

    # Simplify some stuff from file name to avoid excessively long directories.
    if 'point_cloud_' in exp_prefix:
        exp_prefix = exp_prefix.replace('point_cloud_', 'PCL_')
    if 'pointnet_' in exp_prefix:
        exp_prefix = exp_prefix.replace('pointnet_', 'PNet2_')
    return exp_prefix


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=False)
@click.option('--dry/--no-dry', default=False)
@click.option('--num_samples', default='full')
def main(mode, debug, dry, num_samples):
    # ------------------------------------------------------------------------- #
    # Decision 1: pick the correct datasets.
    # ------------------------------------------------------------------------- #


    # ----------------------------- ADJUST -------------------------------- #
    suffix = 'v07_rotation_translation_variably_composed'
    data_info = get_dataset_info(suffix, mode, num_samples)
    # --------------------------------------------------------------------- #

    # ------------------------------------------------------------------------- #
    # Decision 2: pick the method, normally it should be one of:
    #   exp_configs.NAIVE_CLASS_PN2_TO_VECTOR_6DoF  (naive baseline)
    #   exp_configs.SVD_POINTWISE_3D_FLOW  (3D flow and SVD)
    #   exp_configs.SVD_POINTWISE_6D_FLOW  (6D flow and SVD)
    #
    # To debug with translation only, when we regress to 3DoF vectors:
    #   exp_configs.NAIVE_CLASS_PN2_TO_VECTOR_3DoF
    # ------------------------------------------------------------------------- #
    this_cfg = exp_configs.SVD_POINTWISE_3D_FLOW

    # Update config.
    cfg = exp_configs.DEFAULT_CFG
    cfg.update(**this_cfg)
    obs_type = cfg['obs_type']
    act_type = cfg['act_type']
    encoder_type = cfg['encoder_type']
    remove_zeros_PCL = cfg['remove_zeros_PCL']
    method_flow2act = cfg['method_flow2act']
    use_consistency_loss = cfg['use_consistency_loss']
    lambda_consistency = cfg['lambda_consistency']
    lambda_pos = cfg['lambda_pos']
    lambda_rot = cfg['lambda_rot']
    scale_pcl_flow = cfg['scale_pcl_flow']
    scale_pcl_val = cfg['scale_pcl_val']
    scale_targets = cfg['scale_targets']
    use_geodesic_dist = cfg['use_geodesic_dist']
    data_augm_PCL = cfg['data_augm_PCL']
    image_size_crop = cfg['image_size_crop']

    # Some random checks and assertions.
    assert not (scale_pcl_flow and scale_targets), 'Do not do both!'

    # Automatically get exp prefix from settings.
    exp_prefix = get_exp_prefix(
            n_train_demos=data_info['n_train_demos'],
            suffix=suffix,
            obs_type=obs_type,
            encoder_type=encoder_type,
            act_type=act_type,
            scale_pcl_flow=scale_pcl_flow,
            scale_targets=scale_targets,
    )


    # ----------------------------- ADJUST -------------------------------- #

    vg = VariantGenerator()
    vg.add('wandb_project', ['']) # Fill this in!
    vg.add('wandb_entity', ['']) # Fill this in!
    vg.add('data_info', [data_info])  # pass in the full info
    vg.add('image_size_crop', [image_size_crop])
    vg.add('agent', ['bc'])
    vg.add('algorithm', ['BC'])
    vg.add('obs_type', [obs_type])
    vg.add('act_type', [act_type])  # the BC _targets_, not to be confused with `act_mode`
    vg.add('bc_data_dir', [data_info['bc_data_dir']])
    vg.add('encoder_type', [encoder_type])
    vg.add('method_flow2act', [method_flow2act])  # action selection, flow -> (action_type)
    vg.add('hidden_dim', [256])  # for CNNs only
    vg.add('num_layers', [4])  # for CNNs only
    vg.add('num_filters', [32])  # for CNNs only
    vg.add('actor_lr', [1e-4])
    vg.add('n_epochs', [3000])
    vg.add('n_train_demos', [data_info['n_train_demos']])  # new, will subsample if needed
    vg.add('n_valid_demos', [data_info['n_valid_demos']])  # total amount of validation samples
    vg.add('data_buffer_capacity', [1000000]) # keep this at least this large
    vg.add('batch_size', [32])
    vg.add('save_video', [True])
    vg.add('save_model', [True])
    vg.add('load_model', [False])
    vg.add('save_freq', [20])
    vg.add('log_interval', [1])
    vg.add('test_overfitting', [False])  # should normally be False
    vg.add('save_eval_outputs', [False])  # should normally be False
    vg.add('log_flow_visuals', [True])  # probably set as True
    vg.add('scale_pcl_flow', [scale_pcl_flow])
    vg.add('scale_pcl_val', [scale_pcl_val])
    vg.add('scale_targets', [scale_targets])  # careful!
    vg.add('weighted_R_t_loss', [False])
    vg.add('lambda_pos', [lambda_pos])
    vg.add('lambda_rot', [lambda_rot])
    vg.add('use_consistency_loss', [use_consistency_loss])
    vg.add('lambda_consistency', [lambda_consistency])
    vg.add('use_geodesic_dist', [use_geodesic_dist])
    vg.add('data_augm_PCL', [data_augm_PCL])
    vg.add('remove_zeros_PCL', [remove_zeros_PCL])
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
                compile_script = None #'./compile_1.0.sh'  # For the first experiment, compile the current softgym
                wait_compile = None
            else:
                compile_script = None
                wait_compile = 300  # Wait 300 (was originally 30) seconds for compilation to finish
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
