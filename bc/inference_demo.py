import pickle
import time
import zmq
import numpy as np
from pyquaternion import Quaternion

DEMO_DATA = "/data/seita/softgym_mm/data_demo/physicalEnv_v01_rotationsAlso_exp_1/BC_sess7_bc_thisOne_8.pkl"

if __name__ == "__main__":
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

    from zmq import ssh

    print('done with output sock')

    act_socket = context.socket(zmq.SUB)
    act_socket.subscribe("")
    tunnel = ssh.tunnel_connection(act_socket, "tcp://127.0.0.1:5698", "sarthak@omega.rpad.cs.cmu.edu")
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

    # Load demo data
    with open(DEMO_DATA, "rb") as f:
        data = pickle.load(f)
    len_o = len(data['obs'])

    actions = []

    start = time.time()
    for t in range(len_o):
        obs_tuple = data['obs'][t]
        obs = obs_tuple[3]
        # obs = torch.tensor(obs_tuple[3], dtype=torch.float32)
        info = obs_tuple[0]
        # So Python 2 assumes ascii as the default encoding when transacting data, whereas Python 3 assumes UTF-8.
        # twofish runs Python 2 since it needs ROS Kinectic to talk to the Sawyers', so we need to swap the encoding before send it over
        # using zmq
        # In order to change the encoding we first change the object into bytes, and then to UTF-8. Same
        #for the numpy characters and t integer
        obs = np.char.decode(obs.astype(np.bytes_), 'UTF-8')
        info = np.char.decode(info.astype(np.bytes_), 'UTF-8')
        t = str(t).decode("utf-8")
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

        quaternion = Quaternion(axis = [action['action'][3], action['action'][4], action['action'][5]], angle = 1)
        print('Translation: {} Rotation Axis Angle: {} Quaternion: {}'.format(action['action'][:3], action['action'][3:], quaternion))

    end = time.time()
    print("Time elapsed: {}".format(end - start))
    print("Frame rate: {}".format(len_o / (end - start)))