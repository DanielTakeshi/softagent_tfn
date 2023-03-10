import os.path as osp
import os

# TODO change this before make it into a pip package
PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

LOG_DIR = os.path.join(PROJECT_PATH, "data")

# Make sure to use absolute path
REMOTE_DIR = {
    'seuss': '/home/dseita/softagent_rpad_MM',
    'autobot': '/home/xlin3/Projects/softagent',
    'psc': '/home/xlin3/Projects/softagent',
    'nsh': '/home/xingyu/Projects/softagent',
    'yertle': '/home/xingyu/Projects/softagent'
}

REMOTE_MOUNT_OPTION = {
    'seuss': '/usr/share/glvnd',
    'autobot': '/usr/share/glvnd',
    # 'psc': '/pylon5/ir5fpfp/xlin3/Projects/baselines_hrl/:/mnt',
}

REMOTE_LOG_DIR = {
    'seuss': os.path.join(REMOTE_DIR['seuss'], "data"),
    'autobot': os.path.join(REMOTE_DIR['autobot'], "data"),
    # 'psc': os.path.join(REMOTE_DIR['psc'], "data")
    'psc': os.path.join('/mnt', "data"),
}

# PSC: https://www.psc.edu/bridges/user-guide/running-jobs
# partition include [RM, RM-shared, LM, GPU]
# TODO change cpu-per-task based on the actual cpus needed (on psc)
# #SBATCH --exclude=compute-0-[7,11]
# Adding this will make the job to grab the whole gpu. #SBATCH --gres=gpu:1
REMOTE_HEADER = dict(seuss="""
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --exclude=compute-0-[7,9,27]
#SBATCH --cpus-per-task=8
#SBATCH --time=480:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=110G
""".strip(), psc="""
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=RM
#SBATCH --ntasks-per-node=18
#SBATCH --time=48:00:00
#SBATCH --mem=64G
""".strip(), psc_gpu="""
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:p100:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=48:00:00
""".strip(), autobot="""
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=32
#SBATCH --time=3-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
""".strip())

# location of the singularity file related to the project
SIMG_DIR = {
    'seuss': '/home/xlin3/softgym_containers/softgymcontainer_v4.simg',
    'autobot': '/home/xlin3/softgym_containers/softgymcontainer_v3.simg',
    # 'psc': '$SCRATCH/containers/ubuntu-16.04-lts-rl.img',
    'psc': '/pylon5/ir5fpfp/xlin3/containers/ubuntu-16.04-lts-rl.img',

}
CUDA_MODULE = {
    'seuss': 'cuda-91',
    'autobot': 'cuda-10.2',
    'psc': 'cuda/9.0',
}
MODULES = {
    'seuss': ['singularity'],
    'autobot': ['singularity'],
    'psc': ['singularity'],
}
