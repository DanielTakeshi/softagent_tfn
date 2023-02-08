# Behavioral Cloning

Behavioral Cloning, this is hopefully straightforward.
This directory is borrowed from the structure of CURL/SAC here.

**Installation**: follow the instructions from [SoftGym_MM][1] README to create
the conda environment. If running the SoftGym code, you will also need to go
through the Docker pipeline to run it. If just using physical robot code, there
might be a way to avoid this.

## Generating Data for Behavioral Cloning

This is done in **SoftGym**, not SoftAgent, but here are the steps.

1. Run `./bash_scripts/gen_cache_mm.sh` to generate cached data.
2. Run `./bash_scripts/gen_data_bc.sh` to generate demonstration data (edit: now `./bash_scripts/gen_data_bc03.sh`).

The bash scripts contain some extra documentation, such as with the MM env
version, the nature of demonstrations, etc.

Make sure the data is stored in the appropriate file directory. On marvin it is stored here:

```
seita@marvin:/data/seita/softgym_mm/data_demo $ ls -lh
total 103G
drwxrwxr-x 2 seita seita 124K Apr 26 13:50 MMMultiSphere_v02_BClone_filtered_ladle_algorithmic_v02_nVars_2000_obs_combo_act_translation
drwxrwxr-x 2 seita seita 144K Apr 26 20:48 MMMultiSphere_v02_BClone_filtered_ladle_algorithmic_v04_nVars_2000_obs_combo_act_translation_axis_angle
drwxrwxr-x 2 seita seita  60K May 12 00:30 MMMultiSphere_v02_BClone_filtered_ladle_algorithmic_v05_nVars_2000_obs_combo_act_translation
-rw-rw-r-- 1 seita seita  19G May 12 11:15 MMMultiSphere_v02_BClone_filtered_ladle_algorithmic_v05_nVars_2000_obs_combo_act_translation.tar.gz
drwxrwxr-x 2 seita seita  48K May 12 09:54 MMMultiSphere_v02_BClone_filtered_ladle_algorithmic_v06_nVars_2000_obs_combo_act_translation_axis_angle
-rw-rw-r-- 1 seita seita  19G May 12 11:18 MMMultiSphere_v02_BClone_filtered_ladle_algorithmic_v06_nVars_2000_obs_combo_act_translation_axis_angle.tar.gz
drwxrwxr-x 2 seita seita 120K May 13 14:53 MMMultiSphere_v02_BClone_filtered_ladle_algorithmic_v08_nVars_2000_obs_combo_act_translation
-rw-rw-r-- 1 seita seita 7.5G May 13 14:54 MMMultiSphere_v02_BClone_filtered_ladle_algorithmic_v08_nVars_2000_obs_combo_act_translation.tar.gz
drwxrwxr-x 2 seita seita 128K May 19 01:14 MMMultiSphere_v02_BClone_filtered_ladle_algorithmic_v09_nVars_2000_obs_combo_act_translation_axis_angle
-rw-rw-r-- 1 seita seita 7.9G May 19 10:01 MMMultiSphere_v02_BClone_filtered_ladle_algorithmic_v09_nVars_2000_obs_combo_act_translation_axis_angle.tar.gz
drwxrwxr-x 2 seita seita 128K May  2 17:43 MMOneSphere_v01_BClone_filtered_ladle_algorithmic_v02_nVars_2000_obs_combo_act_translation
drwxrwxr-x 2 seita seita 140K May  2 17:44 MMOneSphere_v01_BClone_filtered_ladle_algorithmic_v04_nVars_2000_obs_combo_act_translation_axis_angle
drwxrwxr-x 2 seita seita  60K May 11 20:33 MMOneSphere_v01_BClone_filtered_ladle_algorithmic_v05_nVars_2000_obs_combo_act_translation
-rw-rw-r-- 1 seita seita  18G May 12 11:21 MMOneSphere_v01_BClone_filtered_ladle_algorithmic_v05_nVars_2000_obs_combo_act_translation.tar.gz
drwxrwxr-x 2 seita seita  52K May 12 05:16 MMOneSphere_v01_BClone_filtered_ladle_algorithmic_v06_nVars_2000_obs_combo_act_translation_axis_angle
-rw-rw-r-- 1 seita seita  19G May 12 11:23 MMOneSphere_v01_BClone_filtered_ladle_algorithmic_v06_nVars_2000_obs_combo_act_translation_axis_angle.tar.gz
drwxrwxr-x 2 seita seita 124K May 12 18:56 MMOneSphere_v01_BClone_filtered_ladle_algorithmic_v08_nVars_2000_obs_combo_act_translation
-rw-rw-r-- 1 seita seita 7.2G May 12 20:15 MMOneSphere_v01_BClone_filtered_ladle_algorithmic_v08_nVars_2000_obs_combo_act_translation.tar.gz
drwxrwxr-x 2 seita seita 128K May 17 20:45 MMOneSphere_v01_BClone_filtered_ladle_algorithmic_v09_nVars_2000_obs_combo_act_translation_axis_angle
-rw-rw-r-- 1 seita seita 7.5G May 19 10:03 MMOneSphere_v01_BClone_filtered_ladle_algorithmic_v09_nVars_2000_obs_combo_act_translation_axis_angle.tar.gz
drwxrwxr-x 6 seita seita 4.0K May 12 11:56 old_directories
drwxrwxr-x 2 seita seita 4.0K May 12 11:09 old_tar_gz
seita@marvin:/data/seita/softgym_mm/data_demo $
```

There are some `tar.gz` files since those were faster to `scp` to other machines
than sending each file in its directory.

## Using the code

A few things to check in the data:

- Check which data is being used.
- Check which experiment configuration is being used.

Assuming the data for behavioral cloning is available on the machine, run with
either one of these commands:

```
python experiments/bc/launch_bc_mm.py --debug
python experiments/bc/launch_bc_mm.py
python experiments/bc/launch_bc_mm.py seuss
```

The first command is for debugging and makes it easier to stop the script in
progress. Use the second command to run an "official" experiment and the third
one if running on the cluster.


[1]:https://github.com/mooey5775/softgym_MM/blob/dev_daniel_bc03/README.md