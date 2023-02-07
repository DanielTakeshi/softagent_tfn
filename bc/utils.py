import os
import numpy as np
import gym
from collections import deque
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.util.shape import view_as_windows
from collections import deque
import torch
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import fps
from pyquaternion import Quaternion


class MixedMediaToolReducer:
    TOOL_DATA_PATH = "bc/100_tool_pts.pkl"
    ACTION_LOW  = np.array([ 0, 0, 0, -1, -1, -1])
    ACTION_HIGH = np.array([ 0, 0, 0,  1,  1,  1])
    DEG_TO_RAD = np.pi / 180.
    MAX_ROT_AXIS_ANG = (10. * DEG_TO_RAD)

    def __init__(self, args, action_repeat):
        #assert args.reduce_tool_points
        self.tool_point_num = args.tool_point_num
        self.action_repeat = action_repeat

        self.MAX_ROT_AXIS_ANG /= action_repeat

        with open(self.TOOL_DATA_PATH, 'rb') as f:
            self.all_tool_points = pickle.load(f)

        # Sample tool points
        ratio = self.tool_point_num / 100
        sampled_idxs = fps(self.all_tool_points, ratio=ratio, random_start=False)
        self.tool_points = self.all_tool_points[sampled_idxs].detach().numpy()

        self.rotation = Quaternion()

        # Prep tool points for rotation
        self.vec_mat = np.zeros((self.tool_point_num, 4, 4), dtype=self.tool_points.dtype)
        self.vec_mat[:, 0, 1] = -self.tool_points[:, 0]
        self.vec_mat[:, 0, 2] = -self.tool_points[:, 1]
        self.vec_mat[:, 0, 3] = -self.tool_points[:, 2]

        self.vec_mat[:, 1, 0] = self.tool_points[:, 0]
        self.vec_mat[:, 1, 2] = -self.tool_points[:, 2]
        self.vec_mat[:, 1, 3] = self.tool_points[:, 1]

        self.vec_mat[:, 2, 0] = self.tool_points[:, 1]
        self.vec_mat[:, 2, 1] = self.tool_points[:, 2]
        self.vec_mat[:, 2, 3] = -self.tool_points[:, 0]

        self.vec_mat[:, 3, 0] = self.tool_points[:, 2]
        self.vec_mat[:, 3, 1] = -self.tool_points[:, 1]
        self.vec_mat[:, 3, 2] = self.tool_points[:, 0]

    def reset(self):
        self.rotation = Quaternion()

    def set_axis(self, axis):
        self.rotation = Quaternion(w=axis[3], x=axis[0], y=axis[1], z=axis[2])

    def step(self, act_raw):
        # act_raw: [x, y, z, rx, ry, rz]
        act_clip = np.clip(act_raw, a_min=self.ACTION_LOW, a_max=self.ACTION_HIGH)
        axis = act_clip[3:]

        dtheta = np.linalg.norm(act_clip[3:])
        # TODO(eddieli): Decide if we should bring this up to date with [mixed_media_env.py:594]
        # if dtheta > self.MAX_ROT_AXIS_ANG:
        # dtheta = dtheta * self.MAX_ROT_AXIS_ANG / np.sqrt(3)
        dtheta = min(dtheta, self.MAX_ROT_AXIS_ANG)

        if dtheta == 0:
            axis = np.array([0., -1., 0.])

        axis = axis / np.linalg.norm(axis)

        for i in range(self.action_repeat):
            axis_world = self.rotation.rotate(axis)
            qt_rotate = Quaternion(axis=axis_world, angle=dtheta)
            self.rotation = qt_rotate * self.rotation

    def reduce_tool(self, obs, info):
        tool_idxs = np.where(obs[:, 3] == 1)[0]
        obs_notool = obs[len(tool_idxs):]

        tool_tip_pos = info[:3]

        # Rotate tool points
        global_rotation = self.rotation
        global_rotation._normalise()
        dqp = global_rotation.conjugate.q

        mid = np.matmul(self.vec_mat, dqp)
        mid = np.expand_dims(mid, axis=-1)

        rotated_tool_pts = global_rotation._q_matrix() @ mid
        rotated_tool_pts = rotated_tool_pts[:, 1:, 0]

        rotated_tool_pts += tool_tip_pos

        num_classes = obs.shape[1] - 3
        tool_onehot = np.zeros((self.tool_point_num, num_classes), dtype=obs.dtype)
        tool_onehot[:, 0] = 1

        tool_reduced = np.concatenate((rotated_tool_pts, tool_onehot), axis=1)
        return np.concatenate((tool_reduced, obs_notool), axis=0)


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


def center_crop_image_mb(image, output_size):
    """Assume a minibatch of B items with center cropping."""
    assert len(image.shape) == 4, f'{image.shape} should be Bx3x128x128'
    h, w = image.shape[2:]
    new_h, new_w = output_size, output_size
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    image = image[:, :, top:top + new_h, left:left + new_w]
    return image


def center_crop_image(image, output_size):
    if image.shape[0] == 1:
        image = image[0]
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    image = image[:, top:top + new_h, left:left + new_w]
    return image


def create_pointnet_pyg_data(obs, device):
    """Process data for PointNet++ architectures.

    Need (data.x, data.pos, data.batch, data.ptr). This case assumes
    we have just one observation and have to make a single torch geometric
    data type. We do need `ptr` to be compatible with the minibatch version
    later for forward passes.

    For ONE observation, not a minibatch, we have the shapes:
        data.x     = [N,2] or [N,3], depends on # of segm classes
        data.pos   = [N,3]
        data.batch = [N,]
    There is no need to do unsqueezing.
    """
    assert len(obs.shape) == 2, obs.shape
    data_x     = obs[:, 3:]
    data_pos   = obs[:, 0:3]
    data_batch = torch.zeros(obs.shape[0], dtype=torch.long)
    data_ptr   = torch.as_tensor(np.array([0, obs.shape[0]]), dtype=torch.long)
    obs = Data(
        x=data_x.to(device),
        pos=data_pos.to(device),
        batch=data_batch.to(device),
        ptr=data_ptr.to(device),
    )
    return obs

# -------------------------------------------------- #
# ---------- Point cloud and flow visuals ---------- #
# -------------------------------------------------- #

def pointcloud(
    T_chart_points: np.ndarray, downsample=5, colors=None, scene="scene",
    name=None, pour_water=False,
) -> go.Scatter3d:
    marker_dict = {"size": 4}
    if colors is not None:
        try:
            a = [f"rgb({r}, {g}, {b})" for r, g, b in colors][::downsample]
            marker_dict["color"] = a
        except:
            marker_dict["color"] = colors[::downsample]

    if pour_water:
        x_vals = T_chart_points[0, ::downsample]
    else:
        x_vals = -T_chart_points[0, ::downsample]

    return go.Scatter3d(
        x=x_vals,
        y=T_chart_points[2, ::downsample],
        z=T_chart_points[1, ::downsample],
        mode="markers",
        marker=marker_dict,
        scene=scene,
        name=name,
        hoverlabel=dict(namelength=50),
        showlegend=False,
    )


def _flow_traces_v2(
    pos, flows, sizeref=0.05, scene="scene", flowcolor="red", name="flow",
    pour_water=False,
):
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # Handle 6D flow GT case, present in the pointwise before SVD ablation
    if flows.shape[1] == 6:
        flows[:, :3] += flows[:, 3:]
        flows = flows[:, :3]

    # normalize flows:
    nonzero_flows = (flows == 0.0).all(axis=-1)
    n_pos = pos[~nonzero_flows]
    n_flows = flows[~nonzero_flows]

    n_dest = n_pos + n_flows * sizeref

    for i in range(len(n_pos)):
        if pour_water:
            x_lines.append(n_pos[i][0])
        else:
            x_lines.append(-n_pos[i][0])
        y_lines.append(n_pos[i][2])
        z_lines.append(n_pos[i][1])
        if pour_water:
            x_lines.append(n_dest[i][0])
        else:
            x_lines.append(-n_dest[i][0])
        y_lines.append(n_dest[i][2])
        z_lines.append(n_dest[i][1])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    lines_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        scene=scene,
        line=dict(color=flowcolor, width=10),
        name=name,
        hoverlabel=dict(namelength=50),
        showlegend=False,
    )

    head_trace = go.Scatter3d(
        x=n_dest[:, 0] if pour_water else -n_dest[:, 0],
        y=n_dest[:, 2],
        z=n_dest[:, 1],
        mode="markers",
        marker={"size": 3, "color": "darkred"},
        scene=scene,
        showlegend=False,
    )

    return [lines_trace, head_trace]


def _square_aspect_ratio(x_range, y_range, z_range):
    x_range, y_range, z_range = np.array(x_range), np.array(y_range), np.array(z_range)
    ranges = np.stack((x_range, y_range, z_range), axis=0)
    sizes = np.abs(ranges[:, 1] - ranges[:, 0])
    return sizes / np.min(sizes)


def _3d_scene_fixed(x_range, y_range, z_range):
    ax, ay, az = _square_aspect_ratio(x_range, y_range, z_range)
    scene = dict(
        xaxis=dict(nticks=10, range=x_range),
        yaxis=dict(nticks=10, range=y_range),
        zaxis=dict(nticks=10, range=z_range),
        # aspectratio=dict(x=1, y=1, z=1),
        aspectratio=dict(x=ax, y=ay, z=az),
    )
    return scene


def _3d_scene(data):
    # Create a 3D scene which is a cube w/ equal aspect ratio and fits all the data.

    assert data.shape[1] == 3
    # Find the ranges for visualizing.
    mean = data.mean(axis=0)
    max_x = np.abs(data[:, 0] - mean[0]).max()
    max_y = np.abs(data[:, 2] - mean[2]).max()
    max_z = np.abs(data[:, 1] - mean[1]).max()
    all_max = max(max(max_x, max_y), max_z)
    scene = dict(
        xaxis=dict(nticks=10, range=[mean[0] - all_max, mean[0] + all_max]),
        yaxis=dict(nticks=10, range=[mean[2] + all_max, mean[2] - all_max]),
        zaxis=dict(nticks=10, range=[mean[1] - all_max, mean[1] + all_max]),
        aspectratio=dict(x=1, y=1, z=1),
    )
    return scene


def create_flow_plot(pts, flow, sizeref=2.0, args=None):
    """Create flow plot to show on wandb, current points + (predicted) flow.

    Note: tried numerous ways to add titles and it strangely seems hard. To
    add more info, I'm adjusting the names we supply to the scatter plot and
    increasing its `hoverlabel`. Only for 3D flow!
    """
    # If scaling, then rescale so we can keep the same coordinate scale.
    # Careful if we change the way we do this! See the replay buffer code.
    if args.scale_pcl_flow:
        pts = pts  / args.scale_pcl_val
        flow = flow / args.scale_pcl_val
    elif args.scale_targets:
        # PointNet++ averaging but with scaling of the targets. TODO make more general.
        flow = flow / 250.

    pour_water = args.env_name in ['PourWater', 'PourWater6D']
    if pour_water:
        # Our pointcloud now should include all points adaptively
        scene = _3d_scene(pts)
        sizeref = 10.0
    else:
        scene_3d_vals = ([-0.20, 0.20], [-0.20, 0.20], [0.0, 0.65])
        scene = _3d_scene_fixed(*scene_3d_vals)

    pts_name = f'pts {pts.shape}'
    flow_name = f'{np.mean(flow[:,0]):0.4f},{np.mean(flow[:,1]):0.4f},{np.mean(flow[:,2]):0.4f}'

    # Shrink layout, otherwise we get a lot of whitespace.
    layout = go.Layout(
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )

    f = go.Figure(layout=layout)
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene1", name=pts_name, pour_water=pour_water)
    )
    ts = _flow_traces_v2(pts, flow, sizeref=sizeref, scene="scene1", name=flow_name, pour_water=pour_water)
    for t in ts:
        f.add_trace(t)
    f.update_layout(scene1=scene)

    _adjust_camera(f, pour_water)
    return f


def create_flow_gt_plot(pts, flow, gt, sizeref=2.0, args=None):
    """Create flow plot to show on wandb, current, predicted/gt flow.
    Only for 3D flow!
    """
    pts_name = f'pts {pts.shape}'
    flow_name = f'{np.mean(flow[:,0]):0.4f},{np.mean(flow[:,1]):0.4f},{np.mean(flow[:,2]):0.4f}'
    flow_name_gt = f'{np.mean(gt[:,0]):0.4f},{np.mean(gt[:,1]):0.4f},{np.mean(gt[:,2]):0.4f}'

    # If scaling, then rescale so we can keep the same coordinate scale.
    # Careful if we change the way we do this! See the replay buffer code.
    if args.scale_pcl_flow:
        pts = pts  / args.scale_pcl_val
        flow = flow / args.scale_pcl_val
        gt = gt / args.scale_pcl_val
    elif args.scale_targets:
        # PointNet++ averaging but with scaling of the targets. TODO make more general.
        flow = flow / 250.
        gt = gt / 250.

    pour_water = args.env_name in ['PourWater', 'PourWater6D']
    if pour_water:
        # Our pointcloud now should include all points adaptively
        scene = _3d_scene(pts)
        sizeref = 10.0
    else:
        scene_3d_vals = ([-0.20, 0.20], [-0.20, 0.20], [0.0, 0.65])
        scene = _3d_scene_fixed(*scene_3d_vals)

    f = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
        ],
        subplot_titles=(
            "pred flow",
            "target flow",
        ),
    )

    # Predicted flow plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene1", name=pts_name, pour_water=pour_water)
    )
    ts = _flow_traces_v2(pts, flow, sizeref=sizeref, scene="scene1", name=flow_name, pour_water=pour_water)
    for t in ts:
        f.add_trace(t)
    f.update_layout(
        scene1=scene,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    # Ground truth flow plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene2", name=pts_name, pour_water=pour_water),
        row=1, col=2,
    )
    ts = _flow_traces_v2(pts, gt, sizeref=sizeref, scene="scene2", name=flow_name_gt, pour_water=pour_water)
    for t in ts:
        f.add_trace(t, row=1, col=2)
    f.update_layout(scene2=scene)

    _adjust_camera(f, pour_water)
    return f


def create_6d_flow_gt_plot(pts, flow, flow_r, act, gt, sizeref=2.0, args=None):
    """Create flow plot to show on wandb, current, predicted/gt flow.
    Now supports pouring and scooping."""
    pour_water = args.env_name == 'PourWater'
    if pour_water:
        scene_3d_vals = ([-0.05, 0.75], [-0.20, 0.20], [0.0, 0.65])
        sizeref = 10.0
    else:
        scene_3d_vals = ([-0.20, 0.20], [-0.20, 0.20], [0.0, 0.65])

    pts_name = f'pts {pts.shape}'
    flow_name = f'{np.mean(flow[:,0]):0.4f},{np.mean(flow[:,1]):0.4f},{np.mean(flow[:,2]):0.4f}'
    flow_r_name = f'{np.mean(flow_r[:,0]):0.4f},{np.mean(flow_r[:,1]):0.4f},{np.mean(flow_r[:,2]):0.4f}'
    act_name = f'{np.mean(act[:,0]):0.4f},{np.mean(act[:,1]):0.4f},{np.mean(act[:,2]):0.4f}'
    flow_name_gt = f'{np.mean(gt[:,0]):0.4f},{np.mean(gt[:,1]):0.4f},{np.mean(gt[:,2]):0.4f}'

    # If scaling, then rescale so we can keep the same coordinate scale.
    # Careful if we change the way we do this! See the replay buffer code.
    if args.scale_pcl_flow:
        pts = pts  / args.scale_pcl_val
        flow = flow / args.scale_pcl_val
        flow_r = flow_r / args.scale_pcl_val
        act = act / args.scale_pcl_val
        gt = gt / args.scale_pcl_val
    elif args.scale_targets:
        # PointNet++ averaging but with scaling of the targets. TODO make more general.
        flow = flow / 250.
        flow_r = flow_r / 250.
        act = act / 250.
        gt = gt / 250.

    # Create point colors
    BASE_COLOR = np.array([102, 109, 246])
    colors = np.zeros_like(pts)
    colors += BASE_COLOR

    f = make_subplots(
        rows=2,
        cols=2,
        specs=[
                [{"type": "scene"},
                {"type": "scene"}],
                [{"type": "scene"},
                {"type": "scene"}],
        ],
        subplot_titles=(
            "pred translation",
            "pred rotation",
            "combined flow",
            "target flow",
        ),
    )

    # Predicted translation plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene1", colors=colors, name=pts_name, pour_water=pour_water)
    )
    ts = _flow_traces_v2(pts, flow, sizeref=sizeref, scene="scene1", name=flow_name, pour_water=pour_water)
    for t in ts:
        f.add_trace(t)
    f.update_layout(
        scene1=_3d_scene_fixed(*scene_3d_vals),
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    # Predicted rotation plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene2", colors=colors, name=pts_name, pour_water=pour_water),
        row=1, col=2,
    )
    ts = _flow_traces_v2(pts, flow_r, sizeref=sizeref, scene="scene2", name=flow_r_name, pour_water=pour_water)
    for t in ts:
        f.add_trace(t, row=1, col=2)
    f.update_layout(scene2=_3d_scene_fixed(*scene_3d_vals))

    # Combined flow plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene3", colors=colors, name=pts_name, pour_water=pour_water),
        row=2, col=1,
    )
    ts = _flow_traces_v2(pts, act, sizeref=sizeref, scene="scene3", name=act_name, pour_water=pour_water)
    for t in ts:
        f.add_trace(t, row=2, col=1)
    f.update_layout(scene3=_3d_scene_fixed(*scene_3d_vals))

    # Ground truth flow plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene4", colors=colors, name=pts_name, pour_water=pour_water),
        row=2, col=2,
    )
    ts = _flow_traces_v2(pts, gt, sizeref=sizeref, scene="scene4", name=flow_name_gt, pour_water=pour_water)
    for t in ts:
        f.add_trace(t, row=2, col=2)
    f.update_layout(scene4=_3d_scene_fixed(*scene_3d_vals))

    _adjust_camera(f, pour_water)
    return f


def create_6d_flow_plot(pts, flow, flow_r, act, sizeref=2.0, args=None,
        just_combined=False):
    """Create flow plot to show on wandb, current, predicted flow.
    Now supports pouring and scooping."""
    pour_water = args.env_name == 'PourWater'
    if pour_water:
        scene_3d_vals = ([-0.05, 0.75], [-0.20, 0.20], [0.0, 0.65])
        sizeref = 10.0
    else:
        scene_3d_vals = ([-0.20, 0.20], [-0.20, 0.20], [0.0, 0.65])

    pts_name = f'pts {pts.shape}'
    flow_name = f'{np.mean(flow[:,0]):0.4f},{np.mean(flow[:,1]):0.4f},{np.mean(flow[:,2]):0.4f}'
    flow_r_name = f'{np.mean(flow_r[:,0]):0.4f},{np.mean(flow_r[:,1]):0.4f},{np.mean(flow_r[:,2]):0.4f}'
    act_name = f'{np.mean(act[:,0]):0.4f},{np.mean(act[:,1]):0.4f},{np.mean(act[:,2]):0.4f}'

    # If scaling, then rescale so we can keep the same coordinate scale.
    # Careful if we change the way we do this! See the replay buffer code.
    if args.scale_pcl_flow:
        pts = pts  / args.scale_pcl_val
        flow = flow / args.scale_pcl_val
        flow_r = flow_r / args.scale_pcl_val
        act = act / args.scale_pcl_val
    elif args.scale_targets:
        # PointNet++ averaging but with scaling of the targets. TODO make more general.
        flow = flow / 250.
        flow_r = flow_r / 250.
        act = act / 250.

    # Create point colors
    BASE_COLOR = np.array([102, 109, 246])
    colors = np.zeros_like(pts)
    colors += BASE_COLOR

    # Not usually true but when I load the policy I might just want to see this.
    # And here we can put different parameters as needed for downsampling, etc.
    if just_combined:
        layout = go.Layout(
            autosize=True,
            margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
            width=800,
            height=800,
        )
        f = go.Figure(layout=layout)
        f.add_trace(
            pointcloud(pts.T, downsample=1, scene="scene1", colors=colors, name=pts_name, pour_water=pour_water)
        )
        ts = _flow_traces_v2(
            pts[::2], act[::2], sizeref=sizeref, scene="scene1", name=act_name, pour_water=pour_water
        )
        for t in ts:
            f.add_trace(t)
        f.update_layout(scene1=_3d_scene_fixed(*scene_3d_vals))
        _adjust_camera(f, pour_water)
        return f

    # Otherwise do it the usual way.
    f = make_subplots(
        rows=2,
        cols=2,
        specs=[
                [{"type": "scene"},
                {"type": "scene"}],
                [{"type": "scene"},
                {"type": "scene"}],
        ],
        subplot_titles=(
            "pred translation",
            "pred rotation",
            "combined flow",
            "blank",
        ),
    )

    # Predicted translation plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene1", colors=colors, name=pts_name, pour_water=pour_water)
    )
    ts = _flow_traces_v2(pts, flow, sizeref=sizeref, scene="scene1", name=flow_name, pour_water=pour_water)
    for t in ts:
        f.add_trace(t)
    f.update_layout(
        scene1=_3d_scene_fixed(*scene_3d_vals),
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    # Predicted rotation plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene2", colors=colors, name=pts_name, pour_water=pour_water),
        row=1, col=2,
    )
    ts = _flow_traces_v2(pts, flow_r, sizeref=sizeref, scene="scene2", name=flow_r_name, pour_water=pour_water)
    for t in ts:
        f.add_trace(t, row=1, col=2)
    f.update_layout(scene2=_3d_scene_fixed(*scene_3d_vals))

    # Combined flow plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene3", colors=colors, name=pts_name, pour_water=pour_water),
        row=2, col=1,
    )
    ts = _flow_traces_v2(pts, act, sizeref=sizeref, scene="scene3", name=act_name, pour_water=pour_water)
    for t in ts:
        f.add_trace(t, row=2, col=1)
    f.update_layout(scene3=_3d_scene_fixed(*scene_3d_vals))

    _adjust_camera(f, pour_water)
    return f

# ---------------------------- #
# Flow plots but for PourWater #
# ---------------------------- #

def create_pw_flow_plot_debug(pts, flow, sizeref=10.0, args=None, hang=False,
        enforce_scaling=False, time_step=-1, pidx=-1, center_axes=True):
    """Create flow plot for quick debugging when we add to replay buffer for pouring.

    Mainly to check that the `ee2flow` is working as expected when loading data to
    the replay buffer. Does not enforce scaling of points if this is supposed to be
    before it adds to replay buffer. So, this means the positions of points will be
    in roughly [-1,1] for all xyz components.

    Need to be careful about the axes (mainly for pouring discussion here):
    - In SoftGym, x-axis points towards the target (lower right w.r.t. GIFs), same is
        true for plots (no worries here).
    - In SoftGym, y-axis points up, BUT in plots we have z axis pointing up. Thus for
        _plotting_only_ we swap the y and z axis.
    - In SoftGym, z-axis points to lower left (viewing wrt the GIFs). But in plotly,
        the positive axis points to the upper right (viewing wrt the plots). I reverse
        the ranges of the y axis in the plots so that those values (which represent the
        z values in SoftGym) are aligned when we view the plots.

    Args:
        sizeref: scaling for the flow vectors, for ease of debugging, often 5-10x.
        time_step: the time step in the episode (usually 0 to 99).
        pidx: the pickle index (i.e. the index of the episode).
        dynamic_axes: true if we center axes w.r.t. point cloud center of the tool.
    """
    pts_name = f'pts {pts.shape}'
    flow_name = f'{np.mean(flow[:,0]):0.4f},{np.mean(flow[:,1]):0.4f},{np.mean(flow[:,2]):0.4f}'

    # If scaling, then rescale so we can keep the same coordinate scale.
    # Careful if we change the way we do this! See the replay buffer code.
    if enforce_scaling:
        if args.scale_pcl_flow:
            pts = pts  / args.scale_pcl_val
            flow = flow / args.scale_pcl_val
        elif args.scale_targets:
            # PointNet++ averaging but with scaling of the targets.
            # TODO(daniel) is scale_pcl_val the right arg here?
            flow = flow / args.scale_pcl_val

    # Shrink layout, otherwise we get a lot of whitespace.
    layout = go.Layout(
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )

    f = go.Figure(layout=layout)
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene1", name=pts_name, pour_water=True)
    )
    ts = _flow_traces_v2(
        pts, flow, sizeref=sizeref, scene="scene1", name=flow_name, pour_water=True
    )
    for t in ts:
        f.add_trace(t)

    # Daniel: this does not work, so for `_3d_scene_fixed` I supply [ymax,ymin] :/ ...
    #f['layout']['yaxis']['autorange'] = "reversed"

    # For plots, (a) make axes the same scale, (b) center the tool with `center_axes`.
    delta = 0.5
    if center_axes:
        meanpts = np.mean(pts, axis=0)
        assert len(meanpts) == 3, meanpts
        # Unfortunately `pts` is from SoftGym so the `y` values are z in the plots.
        xmin, xmax = meanpts[0] - (delta/2.), meanpts[0] + (delta/2.)
        ymin, ymax = meanpts[2] - (delta/2.), meanpts[2] + (delta/2.)
        zmin, zmax = meanpts[1] - (delta/2.), meanpts[1] + (delta/2.)
    else:
        xmin, xmax =  0.25,  0.25 + delta
        ymin, ymax = -0.25, -0.25 + delta
        zmin, zmax =  0.15,  0.15 + delta
    f.update_layout(
        # This is the default I used for PourWater but might want to adjust:
        #scene1=_3d_scene_fixed([-0.05, 0.75], [-0.20, 0.20], [0.0, 0.65]),
        scene1=_3d_scene_fixed([xmin, xmax], [ymax, ymin], [zmin, zmax]),
    )

    # Possibly add more here.
    _adjust_camera_angle_pw(f)

    # Handle writing to dir.
    ep_idx = str(pidx).zfill(3)
    flow_viz_dir = f'flow_viz_{ep_idx}'
    #if os.path.exists(flow_viz_dir):
    #    shutil.rmtree(flow_viz_dir)
    if not os.path.exists(flow_viz_dir):
        os.mkdir(flow_viz_dir)
    n_flows = len([x for x in os.listdir(flow_viz_dir) if 'flow_' in x and '.html' in x])
    flowname = f'flow_{str(n_flows).zfill(3)}_ep_{ep_idx}_t_{str(time_step).zfill(3)}.html'
    f.write_html( os.path.join(flow_viz_dir,flowname) )
    if hang:
        print(f'Hanging, look at the flow visuals, see {flow_viz_dir}.')
        import time
        while True:
            time.sleep(1)

# ---------------------------------- #
# To test data augmentation          #
# ---------------------------------- #

def pcl_data_aug_viz(args, pts_raw, pts_aug=None, flow_raw=None, flow_aug=None,
        tool_pts=None, sizeref=1.0):
    """Meant to investigate the data augmentation.

    If tool_pts is not None, we only show flow vectors for the tool points.
    """
    pour_water = args.env_name == 'PourWater'
    if pour_water:
        scene_3d_vals = ([-0.05, 0.75], [-0.20, 0.20], [0.0, 0.65])
    else:
        scene_3d_vals = ([-0.20, 0.20], [-0.20, 0.20], [0.0, 0.65])

    # If scaling, then rescale so we can keep the same coordinate scale.
    # Careful if we change the way we do this! See the replay buffer code.
    if args.scale_pcl_flow:
        pts_raw = pts_raw  / args.scale_pcl_val
        if pts_aug is not None:
            pts_aug = pts_aug  / args.scale_pcl_val
        if flow_raw is not None:
            flow_raw = flow_raw / args.scale_pcl_val
        if flow_aug is not None:
            flow_aug = flow_aug / args.scale_pcl_val

    ## Downsample here if desired.
    #pts_raw = pts_raw[::2]
    #pts_aug = pts_aug[::2]
    #flow_raw = flow_raw[::2]
    #flow_aug = flow_aug[::2]

    # Shrink layout, otherwise we get a lot of whitespace.
    layout = go.Layout(
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )
    f = go.Figure(layout=layout)

    # Add the point cloud before and after augmentation.
    f.add_trace(
        pointcloud(pts_raw.T, downsample=1, scene="scene1", colors='red', pour_water=pour_water)
    )
    if pts_aug is not None:
        f.add_trace(
            pointcloud(pts_aug.T, downsample=1, scene="scene1", colors='blue', pour_water=pour_water)
        )

    # Add flow visualizations.
    if flow_raw is not None:
        if tool_pts is not None:
            pts_raw = pts_raw[tool_pts]
            flow_raw = flow_raw[tool_pts]
        ts = _flow_traces_v2(
            pts_raw, flow_raw, sizeref=sizeref, scene="scene1", flowcolor="darkred", pour_water=pour_water
        )
        for t in ts:
            f.add_trace(t)

    # Add flow visualizations for the data augmented version.
    if flow_aug is not None:
        if tool_pts is not None:
            pts_aug = pts_aug[tool_pts]
            flow_aug = flow_aug[tool_pts]
        ts = _flow_traces_v2(
            pts_aug, flow_aug, sizeref=sizeref, scene="scene1", flowcolor="darkblue", pour_water=pour_water
        )
        for t in ts:
            f.add_trace(t)

    # Camera settings.
    f.update_layout(scene1=_3d_scene_fixed(*scene_3d_vals))
    _adjust_camera(f, pour_water)
    return f

# ---------------------------------- #
# Camera angle (not sure if working) #
# ---------------------------------- #

def _adjust_camera(f, pour_water):
    if pour_water:
        _adjust_camera_angle_pw(f)
    else:
        _adjust_camera_angle(f)


def _adjust_camera_angle(f):
    """Adjust default camera angle if desired.

    For default settings: https://plotly.com/python/3d-camera-controls/
    """
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=1.25, z=0.25)
    )
    f.update_layout(scene_camera=camera)


def _adjust_camera_angle_pw(f):
    """Seems reasonable for PourWater."""
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.10, y=-2.00, z=0.10)
    )
    f.update_layout(scene_camera=camera)


# -------------------------------------------------------------------- #
# Oh, this was from experiments.planet.train but a strange location...
# Let's just move it here.
# -------------------------------------------------------------------- #

def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv
