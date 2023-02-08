"""Daniel's testing, assumes we run `python bc/inference_demo_py3.py` in Python3."""
import os
import pickle
import time
from tkinter import LAST
import zmq
import numpy as np
np.set_printoptions(suppress=True, precision=5, edgeitems=20, linewidth=150)
from pyquaternion import Quaternion
from zmq import ssh
from collections import defaultdict

# Just pick the full dataset and we will load the data within these.
DEMO_DATA = "/data/seita/softgym_mm/data_demo/v02_physicalTranslations_pkls/"
NUM_VALID = 6  # these are the LAST set of episodes, for validation (it's data-dependent)!
BATCH_SIZE = 16  # we do not have a smaller batch at the end w/smaller size
MODEL_PTH = "/data/seita/softagent_mm/BCphy_v02_physicalTranslations_pkls_ntrain_0030_PCL_PNet2_acttype_ee_rawPCL_scaleTarg_debug/BCphy_v02_physicalTranslations_pkls_ntrain_0030_PCL_PNet2_acttype_ee_rawPCL_scaleTarg_debug_2022_06_11_23_10_02_0001/model/ckpt_0010.tar"
assert os.path.exists(DEMO_DATA)
assert os.path.exists(MODEL_PTH)
NPCL = 1200


def load_from_trained_model():
    """Gives us a list of individual (obs,act) pairs.

    One item in the list is a tuple with one obs/act (no minibatches).
    We get the obs.x and obs.pos and horizontally stack.
    """

    # model_tail: 'ckpt_WXYZ.tar'
    model_head, model_tail = os.path.split(MODEL_PTH)
    video_head = model_head.replace('/model', '/video')
    info_dict = model_tail.replace('.tar', '_dict.pkl').replace(
            'ckpt_', 'preds_')
    data_head = os.path.join(video_head, info_dict)

    # This is data stored from validation.
    with open(data_head, 'rb') as fh:
        valid_data = pickle.load(fh)

    # We appended to this each minibatch (usually of size 16).
    n_minibatches = len(valid_data['act_pol'])

    # Store individual observations and actions here.
    individual = []

    # each item in `train_data[key]` is a _list_, of a minibatch's elements
    # e.g., train_data['obs_x][0].shape = (1200*16, 3), act would be (16,3)
    for mb in range(n_minibatches):
        obs_x = valid_data['obs_x'][mb]
        obs_p = valid_data['obs_pos'][mb]
        act_p = valid_data['act_pol'][mb]
        num_items_mb = act_p.shape[0]
        assert num_items_mb == 16, num_items_mb
        for item in range(num_items_mb):
            obs1 = obs_p[item*NPCL: (item+1)*NPCL]
            obs2 = obs_x[item*NPCL: (item+1)*NPCL]
            act = act_p[item]
            obs = np.hstack((obs1, obs2))
            individual.append( (obs, act) )

    # Make sure this number makes sense given the dataset.
    print(f'Done loading, len(individual): {len(individual)}')
    print(f'Shape each: obs: {individual[0][0].shape}, act: {individual[0][1].shape}\n')
    return individual


if __name__ == "__main__":
    individual_old = load_from_trained_model()

    # Start inference by running
    # python -m bc.inference [args]
    # Run `python -m bc.inference -h` for help

    # If you want to start inference programmatically, do something like

    # from bc.inference import Inference
    # inference = Inference(
    #     obs_shape=(2000, 6),
    #     exp_config=EXP_CONFIG DICTIONARY,
    #     model_path=MODEL_PATH,
    # )
    # inference.start()

    # You can use inference.stop() to stop it.

    # Init zmq
    context = zmq.Context()
    obs_socket = context.socket(zmq.PUB)
    obs_socket.setsockopt(zmq.SNDHWM, 0)
    obs_socket.bind("tcp://127.0.0.1:2024")

    # Note! You can send observations and receive actions
    # from two different processes/threads.``
    # e.x. observations arrive at 10 Hz and are sent by the camera process
    # robot.py can just listen to act_socket
    print('done with output sock')

    act_socket = context.socket(zmq.SUB)
    act_socket.subscribe("")
    tunnel = ssh.tunnel_connection(act_socket, "tcp://127.0.0.1:5698", "seita@omega.rpad.cs.cmu.edu")
    # act_socket.connect("ipc://@act_out")

    print('no longer blocked!')

    # Now, we are free to send observations and receive actions
    # Note that observations and actions are not necessarily 1:1
    # You can send observations at any rate to inference
    # If inference starts to fall behind, it starts to drop observations
    # This is why an id should be attached to each observation sent
    # so you can distinguish which observation is attached to which action

    # We organize obs as a dictionary like so:
    # obs = {
    #     "id": unique id (ideally monotonically increasing),
    #     "obs": torch observation tensor (shape: (n_points, 6))
    #         this observation should be the same format as the bc data
    #         it expects [4:6]-dims to be a one-hot vector encoding
    #         (tool, target, distractors)
    #     "info": torch or np array, containing the EE position in the
    #         first 3 items
    # }

    # For some reason, we need to send something to wake up the connection
    obs_socket.send_pyobj("wakeup")
    time.sleep(1)

    # Load just validation data.
    pkl_paths = sorted([
        os.path.join(DEMO_DATA,x) for x in os.listdir(DEMO_DATA)
            if x[-4:] == '.pkl' and 'BC' in x])
    pkl_paths = pkl_paths[-NUM_VALID:]

    # Store the individual (obs,act) pairs as we loaded them.
    individual_now = []
    start = time.time()
    overall_idx = 0

    # Load demo data but do validation only.
    for pkl_pth in pkl_paths:
        with open(pkl_pth, "rb") as f:
            data = pickle.load(f)
        len_o = len(data['obs'])

        for t in range(len_o):
            obs_tuple = data['obs'][0]
            obs = obs_tuple[3]
            info = obs_tuple[0]
            assert obs.shape == (1200,6), obs.shape

            # Create observation dict
            obs_dict = {
                "id": t,
                "obs": obs,
                "info": info,
            }

            # Send observation
            obs_socket.send_pyobj(obs_dict)
            print("Sent obs {}".format(t))

            # Receive action
            action = act_socket.recv_pyobj()
            act = action['action'][:3]
            print(f'action: {act}')
            obs_from_socket = action['obs_again']

            # Now for a given (obs,act) we can compare. Get values from eval in train:
            eval_obs, eval_act = individual_old[overall_idx]  # saved during valid part of train
            eval_act = eval_act / 250.0
            #assert np.allclose(obs, eval_obs, rtol=1e-06), \
            #    f'At {overall_idx} see\n{eval_obs}\nvs\n{obs}'
            #assert np.allclose(obs, obs_from_socket, rtol=1e-06), \
            #    f'At {overall_idx} see\n{obs_from_socket}\nvs\n{obs}'

            # Failing? Bad news: action.
            #assert np.array_equal(eval_obs, obs)
            #assert np.array_equal(obs_from_socket, obs)
            #assert np.allclose(eval_act, act), \
            #    f'At {overall_idx} see\n{eval_act}\nvs\n{act}'

            ## Let's tune the allclose condition.
            #assert np.allclose(eval_act, act, rtol=0.002, atol=0.001), \
            #    f'At {overall_idx} see\n{eval_act}\nvs\n{act}'

            overall_idx += 1
            if overall_idx >= len(individual_old):
                print(f'Exiting now due to {len(individual_old)} obs')
                break

    end = time.time()
    print("Time elapsed: {}".format(end - start))
    print("Frame rate: {}".format(len_o / (end - start)))
    print("len of (obs,act): {}".format(len(individual_now)))
