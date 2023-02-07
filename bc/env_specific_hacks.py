"""Some environment-specific hacks.

In many places of the code we are putting in some conditions based on the env.
We should just do all of that here and call the methods so there's no confusion
or forgetting that this condition must be applied here, etc. If we add another
env or add another version to an env (or really change any of the env) please
check these functions.

This is NON-EXHAUSTIVE. There might be other areas with env names, such as in:

- bc/replay_buffer.py
- bc/train.py (look at observation and action processing).
"""
import numpy as np
import torch
from envs.env import SoftGymEnv
from pyquaternion import Quaternion


def get_state_dim(env_name):
    # Must coordinate with `get_state_info_for_mlp`!!
    if env_name == 'MMOneSphere':
        state_dim = 10
    elif env_name in ['PourWater', 'PourWater6D']:
        state_dim = 22
    else:
        raise ValueError()
    return state_dim


def get_state_info_for_mlp(obs_tuple, env_name, tool_reducer):
    """
    Get state-based info, tool poses + object info.

    See `get_obs()` in the environments (the 'combo' obs mode) for details on how
    to replicate the state based info from the obs.

    We use [tool_reducer], which is init'd and stepped in
    [BehavioralCloningData], to get pose information for the ladle in
    MixedMedia
    """
    keypoints = obs_tuple[0]

    if env_name == 'MMOneSphere':
        # Extract ball position. First 3 are tool tip, so 3:6 are for ball center.
        ball_pos = keypoints[3:6]

        # Get ladle position
        ladle_tip = keypoints[:3]

        # Get ladle rotation from [tool_reducer]
        ladle_pyquat = Quaternion(tool_reducer.rotation)
        ladle_pyquat._normalise()
        ladle_elts = ladle_pyquat.elements
        ladle_quat = np.array([ladle_elts[1], ladle_elts[2], ladle_elts[3], ladle_elts[0]])

        # Mash it all together in state
        # Ball pos (3), ladle tip position (3), ladle quaternion (4)
        #state = np.concatenate((ball_pos, ladle_tip, ladle_quat)) # commenting

        # This is giving assertion errors.
        # # New: actually we should have this information now stored at the end.
        # if len(obs_tuple) == 7:
        #     stored_state = obs_tuple[6]
        #     #assert np.array_equal(stored_state[-4:], ladle_quat), \
        #     #    f'{stored_state[-4:]} {ladle_quat}'
        #     diff_abs = np.sum(np.abs(stored_state[-4:] - ladle_quat))
        #     assert diff_abs < 0.1, \
        #         f'{stored_state[-4:]} {ladle_quat}, diff: {diff_abs}'
        # else:
        #     print(f'Warning, this is outdate data, len: {len(obs_tuple)}')

        # Edit: override the state, now that we explicitly store it.
        state = obs_tuple[6]
        assert np.array_equal(state[:3], ball_pos), f'{state[:3]}, {ball_pos}'
        assert np.array_equal(state[3:6], ladle_tip), f'{state[3:6]}, {ladle_tip}'
        #assert np.array_equal(state[6:10], ladle_quat), f'{state[6:10]}, {ladle_quat}'

    elif env_name in ['PourWater', 'PourWater6D']:
        # I just re-ran the dataset to include 7D tuples where now we get the
        # last part to have the % of water in the controlled and then target cup.
        # For 6D pouring maybe we should just use the % in the target cup.
        # keypoints[:,:3]: positions of all the 10 box centers (tool and target).
        # keypoints[:,6:10]: quaternions of those 10 boxes.

        # Flattening = dim 70.
        #boxes = np.concatenate((keypoints[:,:3], keypoints[:,6:10]), axis=1).flatten()

        # Alternative, dim 30.
        #boxes = keypoints[:,:3].flatten()

        # New, the data should have this (as of 08/26).
        water_info = obs_tuple[6]

        # Water info comes before boxes.
        #state = np.concatenate((water_info, boxes))

        # 1 and 6 are left wall for both, can commpare with right wall.
        # I'm taking distances of all these box centers and taking a norm, this
        # should give all info we need.
        pose_info = keypoints

        box_dims = np.array([
            np.linalg.norm(pose_info[0,:3] - pose_info[1,:3]),  # this will give height info
            np.linalg.norm(pose_info[1,:3] - pose_info[2,:3]),
            np.linalg.norm(pose_info[3,:3] - pose_info[4,:3]),
            np.linalg.norm(pose_info[5,:3] - pose_info[6,:3]),  # this will give height info
            np.linalg.norm(pose_info[6,:3] - pose_info[7,:3]),
            np.linalg.norm(pose_info[8,:3] - pose_info[9,:3]),
        ])

        state = np.concatenate((
            water_info,         # 2D water info
            pose_info[0,:3],    # 3D position
            pose_info[0,6:10],  # 4D quaternion (this is actually the same across boxes)
            pose_info[5,:3],    # 3D position, target box
            pose_info[5,6:10],  # 4D quaternion, target box (well this remains fixed...)
            box_dims,
        ))
        return state

    return state


def get_rgbd_depth_segm_masks(obs_tuple, obs_mode, obs_rgb, env_name):
    """Get the depth-segmentation image.

    MMOneSphere: Need (3,H,W) image.
        0: depth, values between 0 and x, usually x is about 1.
        1: tool mask, values should be 0 or 1 (binary).
        2: item mask, values should be 0 or 1 (binary).
    The mask must follow the same convention as it does during runtime for this
    to be a fair learning algorithm!

    From debugging and saving images, pretty sure this is OK. The only real difference
    from SoftGym is that we have things transposed here so it's (C,H,W).

    For PourWater and PourWater6D, we had to generate a new dataset with the relevant
    information. Hopefully this will still work.
        0: depth
        1: tool (cup we control)
        2: target cup (fixed)
        3: water (approx.)

    Update: now supporting different observation modes as well... and providing the
    RGB observation as well with `obs_rgb`.
    """
    assert obs_mode in ['depth_img', 'depth_segm', 'rgb_segm_masks', 'rgbd_segm_masks']
    assert len(obs_tuple) == 6, len(obs_tuple)
    depth_img = obs_tuple[5]  # (H,W)

    # NOTE! This is important, to get pixels in the same [0,1] scale as others.
    rgb_image = obs_rgb.astype(np.float32) / 255.0

    # Extract segmentation mask.
    segm_img = np.transpose(obs_tuple[2], (2,0,1))  # (k,H,W)

    if env_name == 'MMOneSphere':
        assert segm_img.shape == (5,128,128)  # hard-code for now
        mask_tool = segm_img[0,:,:]  # binary {0,255} image, see segm code
        mask_item = segm_img[4,:,:]  # binary {0,255} image, see segm code

        # Item should occlude the tool (makes it match the PCL).
        idxs_occlude = np.logical_and(mask_tool, mask_item)
        mask_tool[idxs_occlude] = 0

        # Concatenate and form the image. Must do the same during BC!
        mask_tool = mask_tool.astype(np.float32) / 255.0  # has only {0.0, 1.0}
        mask_item = mask_item.astype(np.float32) / 255.0  # has only {0.0, 1,0}

        # Sometimes item (but NOT the tool) can be ENTIRELY occluded.
        assert len(np.unique(mask_item)) in [1,2]
        assert len(np.unique(mask_tool)) == 2

        if obs_mode == 'depth_img':
            w,h = depth_img.shape
            result = np.zeros([3,w,h])
            result[0,:,:] = depth_img
            result[1,:,:] = depth_img
            result[2,:,:] = depth_img
        elif obs_mode == 'depth_segm':
            result = np.concatenate(
                    (depth_img[None,...],
                     mask_tool[None,...],
                     mask_item[None,...]), axis=0)
        elif obs_mode == 'rgb_segm_masks':
            result = np.concatenate(
                    (rgb_image,  # (3,H,W)
                     mask_tool[None,...],
                     mask_item[None,...]), axis=0)
        elif obs_mode == 'rgbd_segm_masks':
            result = np.concatenate(
                    (rgb_image,  # (3,H,W)
                     depth_img[None,...],
                     mask_tool[None,...],
                     mask_item[None,...]), axis=0)
    elif env_name in ['PourWater', 'PourWater6D']:
        # We should probably use the branch that has water in the depth!
        assert segm_img.shape == (4,128,128)  # hard-code for now
        mask_tool  = segm_img[1,:,:]  # binary {0,255} image, TOOL (cup we control)
        mask_targ  = segm_img[2,:,:]  # binary {0,255} image, TARG (fixed target cup)
        mask_water = segm_img[3,:,:]  # binary {0,255} image, WATER, might be overlap

        # Concatenate and form the image. Must do the same during BC!
        mask_tool  = mask_tool.astype(np.float32) / 255.0  # has only {0.0, 1.0}
        mask_targ  = mask_targ.astype(np.float32) / 255.0  # has only {0.0, 1,0}
        mask_water = mask_water.astype(np.float32) / 255.0  # has only {0.0, 1,0}

        if obs_mode == 'depth_img':
            w,h = depth_img.shape
            result = np.zeros([3,w,h])
            result[0,:,:] = depth_img
            result[1,:,:] = depth_img
            result[2,:,:] = depth_img
        elif obs_mode == 'depth_segm':
            result = np.concatenate(
                    (depth_img[None,...],
                     mask_tool[None,...],
                     mask_targ[None,...],
                     mask_water[None,...]), axis=0)
        elif obs_mode == 'rgb_segm_masks':
            result = np.concatenate(
                    (rgb_image,  # (3,H,W)
                     mask_tool[None,...],
                     mask_targ[None,...],
                     mask_water[None,...]), axis=0)
        elif obs_mode == 'rgbd_segm_masks':
            result = np.concatenate(
                    (rgb_image,  # (3,H,W)
                     depth_img[None,...],
                     mask_tool[None,...],
                     mask_targ[None,...],
                     mask_water[None,...]), axis=0)
    else:
        raise ValueError()

    ## Debugging -- looks same as it does in SoftGym, whew. :)
    #import cv2, os
    ##k = len([x for x in os.listdir('tmp') if 'depth_' in x and '.png' in x])
    #k = len([x for x in os.listdir('tmp') if 'color_' in x and '.png' in x])
    #d = 2   # depth could be at 3 if images are 0, 1, 2. Or nothing if rgb_segm_masks :)
    #cv2.imwrite(f'tmp/color_{str(k).zfill(3)}.png', (result[:3,:,:] * 255).astype(np.uint8).transpose(1,2,0))
    ##cv2.imwrite(f'tmp/depth_{str(k).zfill(3)}.png', (result[d,:,:] / np.max(result[d,:,:]) * 255).astype(np.uint8))
    #cv2.imwrite(f'tmp/mask1_{str(k).zfill(3)}.png', (result[d+1,:,:] * 255).astype(np.uint8))
    #cv2.imwrite(f'tmp/mask2_{str(k).zfill(3)}.png', (result[d+2,:,:] * 255).astype(np.uint8))
    #cv2.imwrite(f'tmp/mask3_{str(k).zfill(3)}.png', (result[d+3,:,:] * 255).astype(np.uint8))
    return result


def get_fake_point_dense_tf(obs, info, env_name):
    """When we use dense transformation, get a synthetic point as 1st point.

    This will override the existing point at index 0 in the point cloud.
    Used when loading data into a point cloud based replay buffer.

    Currently using the same code for PourWater and PourWater6D. The `info` is a
    (10,14)-shaped array derived from `get_keypoints()` in the SoftGym env code,
    where the first 5 rows are the controlled cup and the last 5 rows are the
    target cup. Thus, we take row 0 which is the bottom wall of the controlled
    cup, and then the first 3 columns, thus the central (x,y,z) position of that
    wall, which is the center of rotation for both envs. Also, the fake point
    has 1 in the 4th item as it's the tool segmentation class index.

    The obs.shape[1] is a hack since in scooping we may or may not have a 3rd
    segmentation class (if a distractor is here). For pouring we always have
    3 segmentation classes.
    """
    if env_name in ['PourWater', 'PourWater6D']:
        fake_point = [info[0,0], info[0,1], info[0,2], 1., 0.]
    elif env_name in ['MMOneSphere', 'MMMultiSphere', 'SpheresLadle']:
        fake_point = [info[0], info[1], info[2], 1., 0.]
    if obs.shape[1] == 6:
        fake_point.append(0.)
    fake_point_np = np.array(fake_point)
    return fake_point_np


def get_info_when_sampling(env_name, info):
    """During training, we should get `info` when sampling (obs,act).

    Note: this overrides the input info, a torch tensor. First index is the
    minibatch dimension.

    Note: not sure if we should change this for PourWater6D? Here we are taking
    the xyz center of the pyflex shape at index 0 which is the bottom wall of
    the controlled cup (i.e., the tool).
    """
    if env_name == 'PourWater':
        info = info[:, 0, :3]
    elif env_name == 'PourWater6D':
        info = info[:, 0, :3]
    elif env_name in ['MMOneSphere', 'MMMultiSphere', 'SpheresLadle']:
        info = info[:, :3]
    return info


def get_tool_origin_testtime(env, env_name):
    """For test-time inference, get tool origin.

    Copy tool tip to avoid alias issues. Need for dense transformations.
    08/20/2022: PourWater6D to use the same tool origin but with env.glass_z?
    """
    if env_name == 'PourWater':
        tool_origin = np.array([env.glass_x, env.glass_y, 0.])
    elif env_name == 'PourWater6D':
        tool_origin = np.array([env.glass_x, env.glass_y, env.glass_z])
    elif env_name in ['MMOneSphere', 'MMMultiSphere', 'SpheresLadle']:
        tool_origin = np.copy(env.tool_state_tip[0, :3])
    else:
        raise ValueError(env_name)
    return tool_origin


def get_info_shape(env, env_name):
    """Used in bc/train.py.

    I don't think the pouring envs actually use obs_dim_keypt_shape though?
    """
    if env_name in ['PourWater', 'PourWater6D']:
        info_shape = env.obs_dim_keypt_shape
    else:
        info_shape = (env.obs_dim_keypt,)
    return info_shape


def get_episode_part_to_use(env_name, bc_data_dir, ep_len, len_o):
    """Determine for a given episode (of length 100) how much to use for buffer?

    We cannot use all 100 due to not having flow for the next (not available) time
    step. Also, SpheresLadle terminates early. And, we also have that MMOneSphere
    has two versions: v01 where we only take 75%, and v02 where we take all the
    nonzero actions (or we can approximate this by taking just the first 3/5 of
    each episode?

    env_name: string representing env name (not including v01, v02, etc.).
    bc_data_dir: string where BC data is stored.
    ep_len: max episode length, for now 100.
    len_o: actual current episode length (only SpheresLadle terminates early).
    """
    if env_name != 'SpheresLadle':
        assert ep_len == len_o, f'{ep_len} {len_o}'

    if (('ladle_algorithmic_v02_' in bc_data_dir) or
            ('ladle_algorithmic_v04_' in bc_data_dir)):
        # MMOneSphere, v01 (or multi-sphere)
        ep_part_to_use = int(0.75 * ep_len)
    elif 'ladle_6dof_rotations_scoop' in bc_data_dir:
        # MMOneSphere, v02, 6DoF demos
        ep_part_to_use = int(0.6 * ep_len)
    elif env_name == 'SpheresLadle':
        ep_part_to_use = len_o - 1
    elif env_name in ['PourWater', 'PourWater6D']:
        ep_part_to_use = ep_len - 1
    else:
        raise NotImplementedError()
    return ep_part_to_use


def env_scaling_targets(act, act_ub, env_name, env_version, inference=False, load=False):
    """Handle (some) logic when it comes to scaling targets.

    I have a special case for MMOneSphere_v02 since I'm seeing all 3 rotation values as
    very small and I think it would be a lot easier just to predict scaled values into
    [-1,1]. Must be either inference time (executing policy) or loading (to buffer).

    UPDATE 08/24: I think it would now actually be better if we also did this for the
    one-sphere case, it seems to be slightly improving PCL Direct Vector MSE. Let's just
    do that? It still way under-performs ToolFlowNet.

    The reason for this hacky stuff is that act_ub for MMOneSphere is 1 for rotations,
    but 0.004 for translations. See `get_env_act_bounds` here for details.

    Assumes symmetrical action upper / lower bounds! Also applies for PourWater, for
    that we have rotation bounds that are a lot smaller, +/- 0.015.

    UPDATE 09/27: if testing with newer rotation representations, use 3D for transl.,
    but 9D for the rotation (since we convert to those) and I don't think we scale
    rotations. For both envs, scale only the first 3 parts from act_ub[0].
    """
    assert inference or load
    assert not (inference and load)
    assert len(act) in [3,6,9,12,13], act

    # Scale up values to make it easier for networks to train.
    if load:
        if len(act) == 12:
            act[:3] = act[:3] / act_ub[0]
        elif env_name == 'MMOneSphere':
            act = act / 0.004
        else:
            act = act / act_ub

    # Undo the scaling for the purpose of taking env steps (actions).
    if inference:
        if len(act) == 12:
            act[:3] = act[:3] * act_ub[0]
        elif env_name == 'MMOneSphere':
            act = act * 0.004
        else:
            act = act * act_ub

    return act


def get_env_act_bounds(env):
    """The `env` here is the `Env()` created in `train.py`.

    The only function this serves is to help with scaling, and usually for scaling
    the targets related to naive, regress-to-MSE, non-flow methods.

    MMOneSphere:
        Returns (-0.004,0.004) for xyz, (-1,1) for rotations. This means we don't
        do any scaling of the rotations but the xyz will be scaled by 250X, which is
        the same scaling factor used in the flow-based methods. With 250X, the max
        translation is +/- 1. The rotation will be +/- 0.0109 for the MM case, so
        we use a 100X scaling factor on the weights. (Used in CoRL 2022 subm.)

    PourWater:
        Returns (-0.010,0.010) for xyz, (-0.015,0.015) for rotations, so the scaling
        will apply to both. However, here I feel like might want to scale translations
        by 250X instead of 100X, as we did for the MM case? That would make the factor
        the same as compared with flow-based scaling? The rotation is still the same,
        at +/- 0.0087 which divided by 0.015 means 0.58, we could switch this but I
        think this is causing too much effort, and 0.58 would be on par with 0.75,
        which is the max translation magnitude. (Used in CoRL 2022 subm.)
    """
    if not isinstance(env, SoftGymEnv):
        return (None, None)

    if hasattr(env, 'pw_env_version'):
        lb = np.array([-0.004, -0.004, -0.004, -0.015, -0.015, -0.015])
        ub = np.array([ 0.004,  0.004,  0.004,  0.015,  0.015,  0.015])
    if hasattr(env, 'mm_env_version'):
        lb = env._env._wrapped_env.action_space.low  # refers to actual MM env bounds
        ub = env._env._wrapped_env.action_space.high  # refers to actual MM env bounds
    if hasattr(env, 'spheres_env_version'):
        # +/- 0.003 per translation dim, compared to +/- 0.004 for MM.
        lb = env._env._wrapped_env.action_space.low
        ub = env._env._wrapped_env.action_space.high
    return (lb, ub)


def get_env_action_mask(env_name, env_version, device=None):
    """Mask to extract the relevant action parts out of the 6D action space.

    Even though in theory we can support 6D manipulation, we don't have evidence of
    this yet because PourWater only has a 3D action space, and the MM envs have 4D.
    At test time we zero-out unused components. This helps to prevent the agent from
    going out-of-distribution.

    During training, this is used in the forward pass only in two cases: class. PN++
    with predicting 6D pose AND with a pointwise loss, and segm PN++ for the dense
    transformation variant, AND with a pointwise loss. In both cases, we have a 6D
    predicted pose (which we'd extract for the action) but where the loss relies on
    applying the pose to the points. If such a pose has nonzero values at the unused
    DoFs, that can only add noise. We don't do this for SVD because given 3D flow
    vectors, it's not clear how to 'remove rotations' though I can see how removing
    the unused translation for PourWater could help (though we didn't need this to
    get good performance).

    If `device` is None, assume this is used at test time for forward pass.
    If `device` is not None, assume this is used in the PointNet++ forward pass.

    The mask should only be used for 6D actions, but for SpheresLadle we don't use
    any rotations. Let's just use (1,1,1) for now and it should error-check if we
    try and use it.
    """
    if env_name == 'PourWater':
        mask = np.array([1,1,0,0,0,1])
    elif env_name == 'PourWater6D':
        mask = np.array([1,1,1,1,1,1])  # all DoFs :)
    elif env_name == 'MMOneSphere':
        if env_version == 'v01':
            mask = np.array([1,1,1,0,1,0])
        elif env_version == 'v02':
            mask = np.array([1,1,1,1,1,1])  # all DoFs :)
    elif env_name == 'MMMultiSphere':
        mask = np.array([1,1,1,0,1,0])
    elif env_name == 'SpheresLadle':
        mask = np.array([1,1,1])
    else:
        raise ValueError(env_name)

    if device is not None:
        mask = torch.from_numpy(mask).float().to(device)
    return mask


def check_env_version(args, env):
    """Error checks to ensure consistency in file naming.

    Only matters if we're using one of our custom envs with env versions.
    If there's an error check the launch script or the SoftGym env code.
    """
    if not isinstance(env, SoftGymEnv):
        return

    if hasattr(env, 'pw_env_version'):
        assert args.env_version[0] == 'v', args.env_version  # should be vXY
        version_int = int(args.env_version[1:])  # int(XY), remove possible leading 0
        assert env.pw_env_version == version_int, \
            f'Check versions! {env.pw_env_version} vs {version_int}'
    if hasattr(env, 'mm_env_version'):
        assert args.env_version[0] == 'v', args.env_version  # should be vXY
        version_int = int(args.env_version[1:])  # int(XY), remove possible leading 0
        assert env.mm_env_version == version_int, \
            f'Check versions! {env.mm_env_version} vs {version_int}'
    if hasattr(env, 'spheres_env_version'):
        assert args.env_version[0] == 'v', args.env_version  # should be vXY
        version_int = int(args.env_version[1:])  # int(XY), remove possible leading 0
        assert env.spheres_env_version == version_int, \
            f'Check versions! {env.spheres_env_version} vs {version_int}'
