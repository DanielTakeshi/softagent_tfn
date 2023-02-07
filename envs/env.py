"""Handle environment logic for CURL / SAC.
New stuff here is mainly the new environments.
"""
import cv2
import numpy as np
import torch
import gym
import softgym
from gym.spaces import Box
from softgym.envs.pour_water import PourWaterPosControlEnv
from softgym.envs.pour_water_amount import PourWaterAmountPosControlEnv
from softgym.envs.pour_water_6d import PourWater6DEnv
from softgym.envs.pass_water import PassWater1DEnv
from softgym.envs.pass_water_torus import PassWater1DTorusEnv
from softgym.envs.transport_torus import TransportTorus1D
from softgym.envs.rope_flatten_new import RopeFlattenNewEnv
from softgym.envs.cloth_flatten import ClothFlattenEnv
from softgym.envs.cloth_fold import ClothFoldEnv
from softgym.envs.cloth_drop import ClothDropEnv
from softgym.envs.rigid_cloth_fold import RigidClothFoldEnv
from softgym.envs.rigid_cloth_drop import RigidClothDropEnv
from softgym.envs.cloth_fold_crumpled import ClothFoldCrumpledEnv
from softgym.envs.cloth_fold_drop import ClothFoldDropEnv
from softgym.utils.normalized_env import normalize
from softgym.envs.mixed_media_one_sphere import MMOneSphereEnv
from softgym.envs.mixed_media_multi_sphere import MMMultiSphereEnv
from softgym.envs.spheres_env_ladle import SpheresLadleEnv

softgym.register_flex_envs()

GYM_ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2',
            'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2',
            'Walker2d-v2']
CONTROL_SUITE_ENVS = ['cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 'cheetah-run',
                      'ball_in_cup-catch', 'walker-walk']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2}

SOFTGYM_ENVS = ['PourWaterPosControl-v0']

SOFTGYM_CUSTOM_ENVS = {'PassWater': PassWater1DEnv,
                       'PassWaterTorus': PassWater1DTorusEnv,
                       'TransportTorus': TransportTorus1D,
                       'PourWater': PourWaterPosControlEnv,
                       'ClothFlatten': ClothFlattenEnv,
                       'ClothFold': ClothFoldEnv,
                       'ClothFoldCrumpled': ClothFoldCrumpledEnv,
                       'ClothFoldDrop': ClothFoldDropEnv,
                       'ClothDrop': ClothDropEnv,
                       'RigidClothFold': RigidClothFoldEnv,
                       'RigidClothDrop': RigidClothDropEnv,
                       'RopeFlattenNew': RopeFlattenNewEnv,
                       'PourWaterAmount': PourWaterAmountPosControlEnv,
                       'ClothFlattenPPP': ClothFlattenEnv, # Picker pick and place
                       'ClothFoldPPP': ClothFoldEnv}

# New envs.
SOFTGYM_CUSTOM_ENVS['MMOneSphere'] = MMOneSphereEnv
SOFTGYM_CUSTOM_ENVS['MMMultiSphere'] = MMMultiSphereEnv
SOFTGYM_CUSTOM_ENVS['SpheresLadle'] = SpheresLadleEnv
SOFTGYM_CUSTOM_ENVS['PourWater6D'] = PourWater6DEnv


# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(
        0.5)  # Quantise to given bit depth and centre
    observation.add_(torch.rand_like(observation).div_(
        2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(
        np.uint8)


def _images_to_observation(images, bit_depth, image_dim, normalize_observation=True,
        rgbd_imgs=False):
    """Convert images from `env.step()` to torch tensors for deep learning.

    08/18/2022: for the new RGBD setting, stick with float32.

    :images: np.array of shape (720,720,3), to get later. Values bounded between [0,255].
        Edit: actually I now change so the camera_width and camera_height are (128,128) so
        we don't need to resize. For cam_rgb and segm, type of `images` is np.uint8.
    :bit_depth: for quantization/dequantization but we don't seem to use that.
    :image_dim: user-provided to resize `images`, typically 128.
    :normalized_observation: Whether to perform quantization/dequantization. In SAC/CURL
        it seems like this is set to False, so we don't normalize, and the input to the
        policy is thus type torch.uint8.
    :returns: torch tensor of shape (1, n_channels, image_dim, image_dim).
    """
    if normalize_observation or rgbd_imgs:
        dtype = torch.float32
    else:
        dtype = torch.uint8
    if images.shape[0] != image_dim:
        assert images.shape[0] >= image_dim, f'{images.shape} vs {image_dim}'
        images = torch.tensor(cv2.resize(images, (image_dim, image_dim), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1),
                              dtype=dtype)  # Resize and put channel first
    else:
        images = torch.tensor(images.transpose(2, 0, 1), dtype=dtype)  # Resize and put channel first
    if normalize_observation:
        preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
    return images.unsqueeze(dim=0)  # Add batch dimension


class ControlSuiteEnv():
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth, image_dim):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        domain, task = env.split('-')
        self.symbolic = symbolic
        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
        if not symbolic:
            self._env = pixels.Wrapper(self._env)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
            print('Using action repeat %d; recommended action repeat for domain is %d' % (
                action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain]))
        self.bit_depth = bit_depth
        self.image_dim = image_dim

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        if self.symbolic:
            return torch.tensor(np.concatenate(
                [np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0),
                dtype=torch.float32).unsqueeze(dim=0)
        else:
            return _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth, self.image_dim)

    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            state = self._env.step(action)
            reward += state.reward
            self.t += 1  # Increment internal timer
            done = state.last() or self.t == self.max_episode_length

            if done:
                break
        if self.symbolic:
            observation = torch.tensor(np.concatenate(
                [np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0),
                dtype=torch.float32).unsqueeze(dim=0)
        else:
            observation = _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth, self.image_dim)
        return observation, reward, done, {}

    def render(self):
        cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self._env.close()

    @property
    def observation_size(self):
        return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in
                    self._env.observation_spec().values()]) if self.symbolic else (3, self.image_dim, self.image_dim)

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        spec = self._env.action_spec()
        return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))


class GymEnv():
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth, image_dim):
        import gym
        self.symbolic = symbolic
        self._env = gym.make(env)
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth
        self.image_dim = image_dim

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        if self.symbolic:
            return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            return _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth, self.image_dim)

    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            if done:
                break
        if self.symbolic:
            observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            observation = _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth, self.image_dim)
        return observation, reward, done, {}

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.image_dim, self.image_dim, 3), dtype=np.float32)

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] if self.symbolic else (3, self.image_dim, self.image_dim)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())


class SoftGymEnv(object):
    """NOTE(daniel): The actual 'env' used in training code, e.g., CURL / SAC.

    The `self._env = normalize(...)` is what we use for testing random_env in SoftGym.
    This class thus acts as a wrapper to `self._env` which is a descendant of FlexEnv.
    If `symbolic`, then don't use images and use other representations, such as keypoints.

    Adding a new `encoder_type` since we might have things like segmentation as we may
    want to alter the preprocessing. However, it could still be useful to keep the same
    image processing (e.g., if segmentation is binary and we have 0s and 255s only).
    Currently unusued. Also, the encoder class divides images by 255 if using CNNs, we
    don't handle that division here.

    08/18/2022: if using RGBD images, probably easier _not_ to convert to uint8 as that
    will remove the depth info. Keep as torch.float32.
    """

    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth,
                 image_dim, env_kwargs=None, normalize_observation=True, scale_reward=1.0,
                 clip_obs=None, obs_process=None, encoder_type=None):
        self.encoder_type = encoder_type
        if env in SOFTGYM_CUSTOM_ENVS:
            self._env = SOFTGYM_CUSTOM_ENVS[env](**env_kwargs)
        else:
            self._env = gym.make(env)
        self._env = normalize(self._env, scale_reward=scale_reward, clip_obs=clip_obs)
        self.symbolic = symbolic
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth
        self.image_dim = image_dim
        if image_dim is None:
            self.image_dim = image_dim = self._env.observation_space.shape[0]

        # Handle new RGBD image logic and double check things.
        self._use_rgbd = False
        if not self.symbolic:
            self.image_c = self._env.observation_space.shape[-1]
            if env_kwargs['observation_mode'] == 'cam_rgb':
                assert self.image_c == 3, self._env.observation_space.shape
            elif env_kwargs['observation_mode'] == 'cam_rgbd':
                assert self.image_c == 4, self._env.observation_space.shape
                self._use_rgbd = True
            elif env_kwargs['observation_mode'] == 'depth_img':
                assert self.image_c == 3, self._env.observation_space.shape
                self._use_rgbd = True
            elif env_kwargs['observation_mode'] == 'depth_segm':
                # We can actually have 3 or 4 here ... use_rgbd is misleading but it leads to
                # using floats as the correct data type.
                assert self.image_c in [3,4], self._env.observation_space.shape
                self._use_rgbd = True
            elif env_kwargs['observation_mode'] == 'rgb_segm_masks':
                # Same comments as earlier ... :(  5=scooping, 6=pouring.
                assert self.image_c in [5,6], self._env.observation_space.shape
                self._use_rgbd = True
            elif env_kwargs['observation_mode'] == 'rgbd_segm_masks':
                # Same comments as earlier ... :(  6=scooping, 7=pouring.
                assert self.image_c in [6,7], self._env.observation_space.shape
                self._use_rgbd = True
            else:
                raise ValueError(env_kwargs['observation_mode'])

        self.normalize_observation = normalize_observation
        self.obs_process = obs_process

    def reset(self, **kwargs):
        self.t = 0  # Reset internal timer
        obs = self._env.reset(**kwargs)
        if self.symbolic:
            if self.obs_process is None:
                if not isinstance(obs, tuple):
                    return torch.tensor(obs, dtype=torch.float32)
                else:
                    return obs
            else:
                return self.obs_process(obs)
        else:
            return _images_to_observation(obs, self.bit_depth, self.image_dim,
                normalize_observation=self.normalize_observation, rgbd_imgs=self._use_rgbd)

    def step(self, action, **kwargs):
        if not isinstance(action, np.ndarray):
            action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            obs, reward_k, done, info = self._env.step(action, **kwargs)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            # print('t:', self.t, self.max_episode_length, done)
            if self.symbolic:
                if self.obs_process is None:
                    if not isinstance(obs, tuple):
                        obs = torch.tensor(obs, dtype=torch.float32)
                else:
                    obs = self.obs_process(obs)
            else:
                obs = _images_to_observation(obs, self.bit_depth, self.image_dim,
                    normalize_observation=self.normalize_observation, rgbd_imgs=self._use_rgbd)
            if done:
                break
        return obs, reward, done, info

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()

    @property
    def observation_space(self):
        if self.symbolic:
            return self._env.observation_space
        else:
            return Box(low=-np.inf, high=np.inf, shape=(self.image_dim, self.image_dim, self.image_c), dtype=np.float32)

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] if self.symbolic else (self.image_c, self.image_dim, self.image_dim)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())

    def __getattr__(self, name):
        """ Relay unknown attribute access to the wrapped_env. """
        if name == '_env':
            # Prevent recursive call on self._env
            raise AttributeError('_env not initialized yet!')
        return getattr(self._env, name)


def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, image_dim, env_kwargs=None, normalize_observation=True,
        scale_reward=1.0, clip_obs=None, obs_process=None, encoder_type=None):
    """NOTE(daniel) called from training scripts, designed to handle variety of envs,
    and to directly produce torch tensors when querying observations."""
    if env in GYM_ENVS:
        return GymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, image_dim)
    elif env in CONTROL_SUITE_ENVS:
        return ControlSuiteEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, image_dim)
    elif env in SOFTGYM_ENVS or env in SOFTGYM_CUSTOM_ENVS:
        return SoftGymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, image_dim, env_kwargs,
                          normalize_observation=normalize_observation,
                          scale_reward=scale_reward,
                          clip_obs=clip_obs,
                          obs_process=obs_process,
                          encoder_type=encoder_type)
    else:
        raise NotImplementedError


# Wrapper for batching environments together
class EnvBatcher():
    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n = n
        self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
        self.dones = [True] * n

    # Resets every environment and returns observation
    def reset(self):
        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)

    # Steps/resets every environment and returns (observation, reward, done)
    def step(self, actions):
        done_mask = torch.nonzero(torch.tensor(self.dones))[:,
                    0]  # Done mask to blank out observations and zero rewards for previously terminated environments
        observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
        dones = [d or prev_d for d, prev_d in
                 zip(dones, self.dones)]  # Env should remain terminated if previously terminated
        self.dones = dones
        observations, rewards, dones = torch.cat(observations), torch.tensor(rewards,
                                                                             dtype=torch.float32), torch.tensor(dones,
                                                                                                                dtype=torch.uint8)
        observations[done_mask] = 0
        rewards[done_mask] = 0
        return observations, rewards, dones, {}

    def close(self):
        [env.close() for env in self.envs]


class WrapperRlkit(object):
    """ Wrap the image env environment. Use all numpy returns and flatten the observation """

    def __init__(self, env):
        self._env = env

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        return np.array(obs).flatten()

    def step(self, action, **kwargs):
        obs, reward, done, info = self._env.step(action, **kwargs)
        return np.array(obs).flatten(), reward, done, info

    def __getattr__(self, name):
        """ Relay unknown attribute access to the wrapped_env. """
        if name == '_env':
            # Prevent recursive call on self._env
            raise AttributeError('_env not initialized yet!')
        return getattr(self._env, name)

    @property
    def imsize(self):
        return self._env.image_dim
