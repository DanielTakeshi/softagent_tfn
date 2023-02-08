import argparse
from os import makedirs
import zmq
import multiprocessing as mp
import torch
import io
import struct
import numpy as np
from zmq import ssh
from os.path import split, dirname, join, isdir

from bc.utils import create_flow_plot
from bc import exp_configs
from bc.train import make_agent

"""
obs = {
    "id": unique id (ideally monotonically increasing),
    "obs": torch observation tensor (shape: (n_points, 6))
        this observation should be the same format as the bc data
        it expects [4:6]-dims to be a one-hot vector encoding
        (tool, target, distractors)
    "info": torch or np array, containing the EE position in the
        first 3 items
}
"""

def np_to_bytes(arr):
    buf = io.BytesIO()
    np.lib.format.write_array(buf, arr, allow_pickle=False)
    return buf.getvalue()

def bytes_to_np(raw):
    buf = io.BytesIO(raw)
    arr = np.lib.format.read_array(buf, allow_pickle=False)
    buf.close()
    return arr

class InferenceArgs:
    def __init__(self, d):
        self.__dict__ = d


def inference_ps(
    shutdown,
    infd,
    outfd,
    outhwm,
    obs_shape,
    exp_config,
    model_path,
    device,
    scale_factor,
):
    print("[INFO] Initializing agent...")
    # Create args object
    arg_dict = {
        "agent": "bc",
        "algorithm": "BC",
        "hidden_dim": 256,
        "num_layers": 4,
        "num_filters": 32,
        "scale_factor":scale_factor, 
        "load_model": True,
        "load_model_path": model_path,
        "log_interval": 1,
        "weighted_R_t_loss": False,
    }
    arg_dict.update(**exp_config)
    args = InferenceArgs(arg_dict)

    # We assume actions are translation + axis-angle here
    # If not, logic on a higher level than this can handle it
    action_shape = (3,)

    # Make agent
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device,
    )
    agent.train(False)

    # Initialize zmq
    print("[INFO] Starting zmq...")
    context = zmq.Context()
    in_socket = context.socket(zmq.SUB)
    in_socket.setsockopt(zmq.RCVHWM, 0)
    # TODO: THIS NEEDS TO BE ENABLED IN GENERAL! Undo this and the lossless changes
    # Commenting this out for now is fine because our inference code is synchronous.
    # If we make it async, we would want to conflate messages together
    # Lossless code is multipart out of paranoia, which doesn't work with CONFLATE.
    # We can relax this later
    # in_socket.setsockopt(zmq.CONFLATE, 1)

    in_socket.subscribe("")
    #! Introducing the SSH tunnel here to connect to lorax. Remember to add your ssh credentials to the ssh-agent here before running inference.py code.
    tunnel = ssh.tunnel_connection(in_socket, "tcp://127.0.0.1:2024", "sarthak@lorax.pc.cs.cmu.edu")

    context = zmq.Context()
    out_socket = context.socket(zmq.PUB)
    out_socket.setsockopt(zmq.SNDHWM, 0)
    out_socket.bind("tcp://127.0.0.1:5698")

    # Main inference loop
    print("[INFO] Inference ready!")

    # This is where the flow visualizations will be stored
    flow_dir = join(dirname(model_path), 'flow_viz')

    # makdir'ing the directory if it hasn't already been named
    if not isdir(flow_dir):
        makedirs(flow_dir)

    while not shutdown.is_set():
        # payload = in_socket.recv_pyobj()
        # try:
        #     id = payload["id"]
        #     obs = payload["obs"]
        #     #! When obs travels from Python2 to Python3, it needs to be in UTF-8 encoding, We reverse it back to np.float32 here.
        #     obs = obs.astype(np.float32)
        #     obs = torch.tensor(obs, dtype=torch.float32)
        #     info = payload["info"]
        #     #! When info travels from Python2 to Python3, it needs to be in UTF-8 encoding, We reverse it back to np.float32 here.
        #     info = info.astype(np.float32)
        # except:
        #     print("[WARNING] Malformed payload... {}".format(payload))
        #     continue
        payload = in_socket.recv_multipart(copy=True)
        try:
            obs = bytes_to_np(payload[0])

            # Should not be necessary, but just in case
            if args.remove_zeros_PCL:
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

            obs = torch.from_numpy(obs).float()
            info = bytes_to_np(payload[1])
            id = struct.unpack('q', payload[2])[0]
        except:
            print("[WARNING] Malformed payload... {}".format(payload))
            continue
        
        print(f"Processing obs {id}", flush=True)
        assert obs.shape[1] in [4, 5, 6]

        np.save(join(dirname(model_path), 'obs_0_{}.npy').format(id), obs)

        # Handle scaling
        if args.scale_pcl_flow:
            raise ValueError("This shouldn't be happening")
            obs[:, :3] *= args.scale_pcl_val
            info *= args.scale_pcl_val

        # Get action
        with torch.no_grad():
            action = agent.select_action(obs, info=info)

        # Get predicted flow
        # This has to be made into a argparse argument!
        pts_in_pcl = obs.shape[0]
        flow_1 = agent.actor.trunk.flow_per_pt[:pts_in_pcl]
        tool_1 = obs[:pts_in_pcl, 3:]
        # Getting the position of all the points in the same pointcloud
        pos_1 = obs[:pts_in_pcl, :3]

        # The following lines different from the training code is because
        # we do not have pyg structured data anymore
        tool_one = torch.where(tool_1[:, 0] == 1)[0]
        targ_one = torch.where(tool_1[:, 1] == 1)[0]
        xyz = pos_1[tool_one].detach().cpu().numpy()
        xyz_t = pos_1[targ_one].detach().cpu().numpy()
        flow = flow_1[tool_one].detach().cpu().numpy()

        # Actually generating the flow
        eval_fig = create_flow_plot(xyz, xyz_t, flow, args=args)

        # Writing the flow to disc in HTML
        eval_fig.write_html(join(flow_dir, 'flow_viz_payload_id_{}.html'.format(id)))

        # Postprocess action
        # TODO: handle quaternion outputs if our policy does that
        assert len(action) in [3, 6]
        if len(action) == 3:
            action = np.concatenate((action, np.zeros(3)))

        # TODO: handle scale_targets, although that case is unlikely
        if args.scale_pcl_flow:
            action[:3] = action[:3] / args.scale_pcl_val

        if args.scale_targets:
            if 'phy' or 'physical' in model_path:
                print('[INFO] Action before scaling: {}'.format(action[:3]))
                action[:3] = action[:3]/scale_factor # TODO(daniel) make less hacky
                print('[INFO] Action after scaling: {}'.format(action[:3]))

        # Send action
        #! Setting the protocol for the pickle backend here, since Python2 sends pickels with protocol 2
        # out_socket.send_pyobj({
        #     "id": id,
        #     "action": action,
        #     'model': split(model_path)
        # }, protocol = 2)
        id_bytes = struct.pack('q', id)
        action_bytes = np_to_bytes(action)
        model_bytes = model_path.encode()

        out_socket.send_multipart([id_bytes, action_bytes, model_bytes])


class Inference:
    outhwm = 10

    def __init__(
        self,
        obs_shape,
        exp_config,
        model_path,
        scale_factor,
        device="cuda",
        infd="ipc://@pc_in",
        outfd="ipc://@act_out",
    ):
        mp.set_start_method('spawn')

        self.infd = infd
        self.outfd = outfd
        self.obs_shape = obs_shape
        self.exp_config = exp_config
        self.model_path = model_path
        self.device = device
        self.scale_factor = scale_factor

        # Multiprocessing housekeeping
        self.shutdown = mp.Event()
        self.ps = None

        self.psargs = (
            self.shutdown,
            self.infd,
            self.outfd,
            self.outhwm,
            self.obs_shape,
            self.exp_config,
            self.model_path,
            self.device,
            self.scale_factor
        )

    def start(self):
        if self.ps is not None:
            raise RuntimeError("Inference already running!")

        self.ps = mp.Process(target=inference_ps, args=self.psargs)
        self.ps.daemon = True
        self.ps.start()
        print(self)

    def stop(self):
        if self.ps is None:
            raise RuntimeError("Inference not running")

        self.shutdown.set()
        self.ps.join()
        self.ps = None
        self.shutdown.clear()

    def __repr__(self):
        rpr = "-------PN++ Inference-------\n"
        rpr += f"IN: {self.infd}\n"
        rpr += f"OUT: {self.outfd}\n"

        return rpr

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    
    # Required arguments
    ap.add_argument("--exp_config", type=str, required=True, help="name of BC exp config")
    ap.add_argument("--model_path", type=str, required=True, help="path to saved checkpoint")

    # Optional arguments
    ap.add_argument("--obs_dim", type=int, default=5, help="dimension of observation")
    ap.add_argument("--max_n_points", type=int, default=1400, help="max number of points")
    ap.add_argument("--scale_factor", type=float, default=250.0, help="max number of points")
    ap.add_argument("--infd", type=str, default="tcp://lorax.pc.cs.cmu.edu:2023", help="incoming inference socket")
    ap.add_argument("--outfd", type=str, default="tcp://127.0.0.1:5698", help="outgoing inference socket")
    ap.add_argument("--device", type=str, default="cuda", help="device to run inference on")
    args = ap.parse_args()

    # Placing these checks to make sure we run inference with the correct configuration
    assert not ('backup' in args.model_path and args.max_n_points == 1400), 'Check max_n_points! Spawining BC agent with {} points, when model is 1200*5/6 shaped'.format(args.max_n_points)
    assert not ('denser' in args.model_path and args.max_n_points == 1200), 'Check max_n_points! Spawining BC agent with {} points, when model is 1400*5/6 shaped'.format(args.max_n_points)
    assert not ('demonstrator' in args.model_path and args.max_n_points == 1200), 'Check max_n_points! Spawining BC agent with {} points, when model is 1400*5/6 shaped'.format(args.max_n_points)

    assert not ('v06' in args.model_path and 'noScaleTarg' not in args.model_path and args.scale_factor != 100.0), 'v06 demonstrators should run with a scaling of 100.0! You have {}'.format(args.scale_factor)
    assert not ('v06' in args.model_path and args.obs_dim != 5), 'v06 demonstrators should have obs_dim of 5! You have {}'.format(args.obs_dim)

    assert not ('v05' in args.model_path and 'noScaleTarg' not in args.model_path and args.scale_factor != 50.0), 'v05 demonstrators should run with a scaling of 50.0! You have {}'.format(args.scale_factor)
    assert not ('v05' in args.model_path and args.obs_dim != 5), 'v05 demonstrators should have obs_dim of 5! You have {}'.format(args.obs_dim)

    assert not ('v04' in args.model_path and 'noScaleTarg' not in args.model_path and args.scale_factor != 250.0), 'v04 demonstrators should run with a scaling of 250.0! You have {}'.format(args.scale_factor)
    assert not ('v04' in args.model_path and args.obs_dim != 5), 'v04 demonstrators should run with a scaling of 5! You have {}'.format(args.obs_dim)

    assert not ('v03' in args.model_path and 'noScaleTarg' not in args.model_path and args.scale_factor != 250.0), 'v03 demonstrators should have obs_dim of 6! You have {}'.format(args.scale_factor)
    assert not ('v03' in args.model_path and args.obs_dim != 6), 'v03 demonstrators should have obs_dim of 6! You have {}'.format(args.obs_dim)

    #! What needs to go in here?
    # 1. Make a list of configurations. In our case the naive and the flow based policy
    # 2. Have a random runmber which picks between 0 and 1
    # 3. Pick that config and load it into the next step. That's it for here

    #! This will pick the index of the config that the script will pick and then execute

    # print('starting with mode: {} and path: {}'.format(args.exp_config, args.model_path))

    # configs = ['SVD_POINTWISE_3D_FLOW', 'NAIVE_CLASS_PN2_TO_VECTOR_3DoF']
    # idx = np.random.choice([0, 1])

    # if idx == 0:
    #     args.model_path = 'data/local/testing_models/flow_ckpt_0500.tar'
    #     args.exp_config = 'SVD_POINTWISE_3D_FLOW'
    # else:
    #     args.model_path = 'data/local/testing_models/naive_ckpt_0300.tar'
    #     args.exp_config = 'NAIVE_CLASS_PN2_TO_VECTOR_3DoF'

    print('model: {} and path: {}'.format(args.exp_config, args.model_path))

    # Load exp config
    exp_config = exp_configs.DEFAULT_CFG
    exp_config.update(**exp_configs.__dict__[args.exp_config])

    assert not (exp_config['scale_targets'] == True and 'noScaleTarg' in args.model_path), 'You have scale_targets on but the model is noScaleTarg: {}'.format(args.model_path)
    assert not (exp_config['scale_targets'] == False and 'scaleTarg' in args.model_path), 'You have scale_targets off but the model is scaleTarg: {}'.format(args.model_path)

    # Create inference object
    inference = Inference(
        obs_shape=(args.max_n_points, args.obs_dim),
        exp_config=exp_config,
        model_path=args.model_path,
        scale_factor = args.scale_factor,
        infd=args.infd,
        outfd=args.outfd,
    )

    # Start inference
    inference.start()

    # Wait for KeyboardInterrupt
    try:
        while True:
            input()
    except KeyboardInterrupt:
        print("Stopping inference...")
        inference.stop()
