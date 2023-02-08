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
from torch_geometric.data import Data
import matplotlib.pyplot as plt


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


# ---------------------------------------------------------------- #
# Point cloud and flow visualizations
# ---------------------------------------------------------------- #

def pointcloud(
    T_chart_points: np.ndarray, downsample=5, colors=None, scene="scene", name=None
) -> go.Scatter3d:
    marker_dict = {"size": 3}
    if colors is not None:
        try:
            a = [f"rgb({r}, {g}, {b})" for r, g, b in colors][::downsample]
            marker_dict["color"] = a
        except:
            marker_dict["color"] = colors[::downsample]
    return go.Scatter3d(
        x=-T_chart_points[0, ::downsample],
        y=T_chart_points[1, ::downsample],
        z=T_chart_points[2, ::downsample],
        mode="markers",
        marker=marker_dict,
        scene=scene,
        name=name,
        hoverlabel=dict(namelength=50),
        showlegend=False,
    )


def _flow_traces_v2(
    pos, flows, sizeref=0.05, scene="scene", flowcolor="red", name="flow"
):
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # normalize flows:
    nonzero_flows = (flows == 0.0).all(axis=-1)
    n_pos = pos[~nonzero_flows]
    n_flows = flows[~nonzero_flows]

    n_dest = n_pos + n_flows * sizeref

    for i in range(len(n_pos)):
        x_lines.append(-n_pos[i][0])
        y_lines.append(n_pos[i][1])
        z_lines.append(n_pos[i][2])
        x_lines.append(-n_dest[i][0])
        y_lines.append(n_dest[i][1])
        z_lines.append(n_dest[i][2])
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
        x=-n_dest[:, 0],
        y=n_dest[:, 1],
        z=n_dest[:, 2],
        mode="markers",
        marker={"size": 3, "color": "darkred"},
        scene=scene,
        showlegend=False,
    )

    return [lines_trace, head_trace]


def _3d_scene_fixed(x_range, y_range, z_range):
    scene = dict(
        xaxis=dict(nticks=10, range=x_range),
        yaxis=dict(nticks=10, range=y_range),
        zaxis=dict(nticks=10, range=z_range),
        aspectratio=dict(x=1, y=1, z=1),
    )
    return scene


def create_flow_plot(pts, pts_T, flow, sizeref=2.0, args=None):
    """Create flow plot to show on wandb, current points + (predicted) flow.

    Note: tried numerous ways to add titles and it strangely seems hard. To
    add more info, I'm adjusting the names we supply to the scatter plot and
    increasing its `hoverlabel`.
    """
    pts_name = f'pts {pts.shape}'
    flow_name = f'{np.mean(flow[:,0]):0.4f},{np.mean(flow[:,1]):0.4f},{np.mean(flow[:,2]):0.4f}'

    # If scaling, then rescale so we can keep the same coordinate scale.
    # Careful if we change the way we do this! See the replay buffer code.
    if args.scale_pcl_flow:
        pts = pts  / args.scale_pcl_val
        flow = flow / args.scale_pcl_val
    elif args.scale_targets:
        # PointNet++ averaging but with scaling of the targets. TODO make more general.
        flow = flow / args.scale_factor

    # Shrink layout, otherwise we get a lot of whitespace.
    layout = go.Layout(
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
    )

    f = go.Figure(layout=layout)
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene1", name=pts_name)
    )
    f.add_trace(
        pointcloud(pts_T.T, downsample=1, scene="scene1", name='targ_points')
    )
    ts = _flow_traces_v2(pts, flow, sizeref=sizeref, scene="scene1", name=flow_name)
    for t in ts:
        f.add_trace(t)
    # f.update_layout(scene1=_3d_scene(sample['points']))
    f.update_layout(
        scene1=_3d_scene_fixed([-1, 0], [-0.6, 0.4], [-0.5, 0.5])
        )

    # NOTE(daniel) IDK why this is not working? Would help to show more info.
    f.update_layout(title_text="Flow Plot", title_font_size=10)

    _adjust_camera_angle(f)
    return f


def create_flow_gt_plot(pts, pts_t, flow, gt, sizeref=2.0, args=None):
    """Create flow plot to show on wandb, current, predicted/gt flow."""
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
        flow = flow * args.data_info['scaling_factor']
        gt = gt * args.data_info['scaling_factor']

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
        pointcloud(pts.T, downsample=1, scene="scene1", name=pts_name)
    )
    f.add_trace(
        pointcloud(pts_t.T, downsample=1, scene="scene1", name='targ_points')
    )
    ts = _flow_traces_v2(pts, flow, sizeref=sizeref, scene="scene1", name=flow_name)
    for t in ts:
        f.add_trace(t)

    f.update_layout(
        scene1=_3d_scene_fixed([-1, 0], [-0.6, 0.4], [-0.5, 0.5]),
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    # Ground truth flow plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene2", name=pts_name),
        row=1, col=2,
    )
    f.add_trace(
        pointcloud(pts_t.T, downsample=1, scene="scene2", name='targ_points'),
        row=1, col=2,
    )
    ts = _flow_traces_v2(pts, gt, sizeref=sizeref, scene="scene2", name=flow_name_gt)
    for t in ts:
        f.add_trace(t, row=1, col=2)

    f.update_layout(scene2=_3d_scene_fixed([-1, 0], [-0.6, 0.4], [-0.5, 0.5]))

    _adjust_camera_angle(f)
    return f


def create_6d_flow_gt_plot(pts, flow, flow_r, act, gt, sizeref=2.0, args=None):
    """Create flow plot to show on wandb, current, predicted/gt flow."""
    pts_name = f'pts {pts.shape}'
    flow_name = f'{np.mean(flow[:,0]):0.4f},{np.mean(flow[:,1]):0.4f},{np.mean(flow[:,2]):0.4f}'
    flow_r_name = f'{np.mean(flow_r[:,0]):0.4f},{np.mean(flow_r[:,1]):0.4f},{np.mean(flow_r[:,2]):0.4f}'
    act_name = f'{np.mean(act[:,0]):0.4f},{np.mean(act[:,1]):0.4f},{np.mean(act[:,2]):0.4f}'
    flow_name_gt = f'{np.mean(gt[:,0]):0.4f},{np.mean(gt[:,1]):0.4f},{np.mean(gt[:,2]):0.4f}'


    print('[SARTHAK] utils? utils')

    print('Pointcloud Shape: {} Flow Shape: {}'.format(pts.shape, flow.shape))

    pts[:, 0] *= -1
    flow[:, 0] *= -1
    pts[:,  [1, 2]] = pts[:, [2, 1]]
    flow[:, [1, 2]] = flow[:, [2, 1]]

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
        pointcloud(pts.T, downsample=1, scene="scene1", colors=colors, name=pts_name)
    )
    ts = _flow_traces_v2(pts, flow, sizeref=sizeref, scene="scene1", name=flow_name)
    for t in ts:
        f.add_trace(t)

    # f.update_layout(scene1=_3d_scene_fixed([-1, 1], [-1, 1], [-1, 1]))

    f.update_layout(
        scene1=_3d_scene_fixed([-1, 1], [-1, 1], [-1, 1]),
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    # Predicted rotation plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene2", colors=colors, name=pts_name),
        row=1, col=2,
    )
    ts = _flow_traces_v2(pts, flow_r, sizeref=sizeref, scene="scene2", name=flow_r_name)
    for t in ts:
        f.add_trace(t, row=1, col=2)
    f.update_layout(scene2=_3d_scene_fixed([-1, 1], [-1, 1], [-1, 1]))
    # f.update_layout(scene2=_3d_scene_fixed([-0.20, 0.20], [-0.20, 0.20], [0.0, 0.65]))

    # Combined flow plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene3", colors=colors, name=pts_name),
        row=2, col=1,
    )
    ts = _flow_traces_v2(pts, act, sizeref=sizeref, scene="scene3", name=act_name)
    for t in ts:
        f.add_trace(t, row=2, col=1)

    f.update_layout(scene3=_3d_scene_fixed([-1, 1], [-1, 1], [-1, 1]))
    # f.update_layout(scene3=_3d_scene_fixed([-0.20, 0.20], [-0.20, 0.20], [0.0, 0.65]))

    # Ground truth flow plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene4", colors=colors, name=pts_name),
        row=2, col=2,
    )
    ts = _flow_traces_v2(pts, gt, sizeref=sizeref, scene="scene4", name=flow_name_gt)
    for t in ts:
        f.add_trace(t, row=2, col=2)

    f.update_layout(scene4=_3d_scene_fixed([-1, 1], [-1, 1], [-1, 1]))

    _adjust_camera_angle(f)
    return f


def create_6d_flow_plot(pts, flow, flow_r, act, sizeref=2.0, args=None):
    """Create flow plot to show on wandb, current, predicted/gt flow."""
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
        pointcloud(pts.T, downsample=1, scene="scene1", colors=colors, name=pts_name)
    )
    ts = _flow_traces_v2(pts, flow, sizeref=sizeref, scene="scene1", name=flow_name)
    for t in ts:
        f.add_trace(t)

    f.update_layout(
        scene1=_3d_scene_fixed([-1, 1], [-1, 1], [-1, 1]),
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    # Predicted rotation plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene2", colors=colors, name=pts_name),
        row=1, col=2,
    )
    ts = _flow_traces_v2(pts, flow_r, sizeref=sizeref, scene="scene2", name=flow_r_name)
    for t in ts:
        f.add_trace(t, row=1, col=2)

    f.update_layout(scene2=_3d_scene_fixed([-1, 1], [-1, 1], [-1, 1]))

    # Combined flow plot
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene3", colors=colors, name=pts_name),
        row=2, col=1,
    )
    ts = _flow_traces_v2(pts, act, sizeref=sizeref, scene="scene3", name=act_name)
    for t in ts:
        f.add_trace(t, row=2, col=1)

    f.update_layout(scene3=_3d_scene_fixed([-1, 1], [-1, 1], [-1, 1]))

    _adjust_camera_angle(f)
    return f


def _adjust_camera_angle(f):
    """Adjust default camera angle if desired.

    For default settings: https://plotly.com/python/3d-camera-controls/
    """
    # camera = dict(
    #     up=dict(x=0, y=0, z=1),
    #     center=dict(x=0, y=0, z=0),
    #     eye=dict(x=1.25, y=1.25, z=0.25)
    # )
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.0, y=-1.0, z=0.5)
    )
    f.update_scenes(camera=camera)


# ---------------------------------------------------------------- #
# Visualize data augmentation as needed; possibly some repetition to
# avoid breaking existing code.
# ---------------------------------------------------------------- #

def pointcloud_real(
    T_chart_points: np.ndarray, downsample=5, colors=None, scene="scene", name=None
) -> go.Scatter3d:
    """Physical data / experiments."""
    marker_dict = {"size": 4}
    if colors is not None:
        try:
            a = [f"rgb({r}, {g}, {b})" for r, g, b in colors][::downsample]
            marker_dict["color"] = a
        except:
            marker_dict["color"] = colors[::downsample]
    return go.Scatter3d(
        x=T_chart_points[0, ::downsample],
        y=T_chart_points[1, ::downsample],
        z=T_chart_points[2, ::downsample],
        mode="markers",
        marker=marker_dict,
        scene=scene,
        name=name,
        hoverlabel=dict(namelength=50),
        showlegend=False,
    )


def flow_traces_real(
    pos, flows, sizeref=0.05, scene="scene", flowcolor="red", name="flow"
):
    """Physical data / experiments.

    Marker size is 3, so the flow + the ending of the flow will be a bit smaller
    than the original source point cloud (marker size 4) which may help visuals.
    """
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # normalize flows:
    nonzero_flows = (flows == 0.0).all(axis=-1)
    n_pos = pos[~nonzero_flows]
    n_flows = flows[~nonzero_flows]

    n_dest = n_pos + n_flows * sizeref

    for i in range(len(n_pos)):
        x_lines.append(n_pos[i][0])
        y_lines.append(n_pos[i][1])
        z_lines.append(n_pos[i][2])
        x_lines.append(n_dest[i][0])
        y_lines.append(n_dest[i][1])
        z_lines.append(n_dest[i][2])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
    lines_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        scene=scene,
        line=dict(color=flowcolor, width=6),
        name=name,
        hoverlabel=dict(namelength=50),
        showlegend=False,
    )

    head_trace = go.Scatter3d(
        x=n_dest[:, 0],
        y=n_dest[:, 1],
        z=n_dest[:, 2],
        mode="markers",
        marker={"size": 3, "color": flowcolor},
        scene=scene,
        showlegend=False,
    )

    return [lines_trace, head_trace]


def pcl_data_aug_viz(pts_raw, pts_aug=None, flow_raw=None, flow_aug=None,
        tool_pts=None, sizeref=1.0):
    """Meant for the physical setup, so some ranges will vary.

    If tool_pts is not None, we only show flow vectors for the tool points.
    """

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
        pointcloud_real(pts_raw.T, downsample=1, scene="scene1", colors='red')
    )
    if pts_aug is not None:
        f.add_trace(
            pointcloud_real(pts_aug.T, downsample=1, scene="scene1", colors='blue')
        )

    # Add flow visualizations.
    if flow_raw is not None:
        if tool_pts is not None:
            pts_raw = pts_raw[tool_pts]
            flow_raw = flow_raw[tool_pts]
        ts = flow_traces_real(
            pts_raw, flow_raw, sizeref=sizeref, scene="scene1", flowcolor="darkred"
        )
        for t in ts:
            f.add_trace(t)

    # Add flow visualizations for the data augmented version.
    if flow_aug is not None:
        if tool_pts is not None:
            pts_aug = pts_aug[tool_pts]
            flow_aug = flow_aug[tool_pts]
        ts = flow_traces_real(
            pts_aug, flow_aug, sizeref=sizeref, scene="scene1", flowcolor="darkblue"
        )
        for t in ts:
            f.add_trace(t)

    # Camera settings.
    f.update_layout(
        # The scene that we use in real.
        scene1=_3d_scene_fixed([0.35, 0.85], [-0.50, 0.00], [-0.10, 0.40])
        # Zero centered.
        #scene1=_3d_scene_fixed([-0.9, 0.9], [-0.9, 0.9], [-0.9, 0.9])
    )
    #_adjust_camera_angle(f)
    return f

# -------------------------------------------------------------------- #
# Data visualizations, e.g. from replay buffer or network predictions
# -------------------------------------------------------------------- #

def plot_action_hist_buffer(acts, suffix, scaling=False):
    """Plots histogram of the delta (x,y,z) translation values in data.

    The network should be predicting values in these ranges at test time.
    Also does rotations as well, assuming `acts.shape` is 6.

    Note: this will only work if our actions are interpreted as 6D translation
    and rotations, not if using flow.
    """
    delta_x = acts[:,0]
    delta_y = acts[:,1]
    delta_z = acts[:,2]
    str_x = f'$\Delta$x, ({np.min(delta_x):0.4f}, {np.max(delta_x):0.4f})'
    str_y = f'$\Delta$y, ({np.min(delta_y):0.4f}, {np.max(delta_y):0.4f})'
    str_z = f'$\Delta$z, ({np.min(delta_z):0.4f}, {np.max(delta_z):0.4f})'
    l_x = f'x, {np.mean(delta_x):0.4f} +/- {np.std(delta_x):0.4f}'
    l_y = f'y, {np.mean(delta_y):0.4f} +/- {np.std(delta_y):0.4f}'
    l_z = f'z, {np.mean(delta_z):0.4f} +/- {np.std(delta_z):0.4f}'

    # Are we also plotting rotations?
    plot_rots = acts.shape[1] == 6
    if plot_rots:
        nrows, ncols = 2, 3
        drot_x = acts[:,3]
        drot_y = acts[:,4]
        drot_z = acts[:,5]
        str_drot_x = f'$\Delta$rotx, ({np.min(drot_x):0.4f}, {np.max(drot_x):0.4f})'
        str_drot_y = f'$\Delta$roty, ({np.min(drot_y):0.4f}, {np.max(drot_y):0.4f})'
        str_drot_z = f'$\Delta$rotz, ({np.min(drot_z):0.4f}, {np.max(drot_z):0.4f})'
        l_drot_x = f'x, {np.mean(drot_x):0.4f} +/- {np.std(drot_x):0.4f}'
        l_drot_y = f'y, {np.mean(drot_y):0.4f} +/- {np.std(drot_y):0.4f}'
        l_drot_z = f'z, {np.mean(drot_z):0.4f} +/- {np.std(drot_z):0.4f}'
    else:
        nrows, ncols = 1, 3

    # Scaling the ranges. Original ranges were set assuming that we had divided
    # by 0.004 (or multiplied by 250) so just undo that here.
    factor = 1.
    if not scaling:
        factor = 250

    # Bells and whistles.
    figname = f'fig_actions_{suffix}_scaling_{scaling}.png'
    titlesize = 32
    ticksize = 28
    legendsize = 23

    _, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(11*ncols, 7*nrows))
    ax[0,0].set_title(str_x, size=titlesize)
    ax[0,1].set_title(str_y, size=titlesize)
    ax[0,2].set_title(str_z, size=titlesize)
    ax[0,0].hist(delta_x, label=l_x, bins=20, edgecolor='k')
    ax[0,1].hist(delta_y, label=l_y, bins=20, edgecolor='k')
    ax[0,2].hist(delta_z, label=l_z, bins=20, edgecolor='k')
    ax[0,0].set_xlim([-4/factor, 4/factor])
    ax[0,1].set_xlim([-4/factor, 4/factor])
    ax[0,2].set_xlim([-5/factor, 5/factor])
    #ax[0,0].set_yscale('log')
    #ax[0,1].set_yscale('log')
    #ax[0,2].set_yscale('log')

    if plot_rots:
        ax[1,0].set_title(str_drot_x, size=titlesize)
        ax[1,1].set_title(str_drot_y, size=titlesize)
        ax[1,2].set_title(str_drot_z, size=titlesize)
        ax[1,0].hist(drot_x, label=l_drot_x, bins=20, edgecolor='k')
        ax[1,1].hist(drot_y, label=l_drot_y, bins=20, edgecolor='k')
        ax[1,2].hist(drot_z, label=l_drot_z, bins=20, edgecolor='k')
        ax[1,0].set_xlim([-0.1/factor, 0.1/factor])
        ax[1,1].set_xlim([-0.1/factor, 0.1/factor])
        ax[1,2].set_xlim([-0.1/factor, 0.1/factor])
        #ax[1,0].set_yscale('log')
        #ax[1,1].set_yscale('log')
        #ax[1,2].set_yscale('log')

    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    plt.savefig(figname)

    print(f'[replay buffer] See plot: {figname}')


def plot_action_net_preds(acts, figname, scaling=False):
    """Plots histogram of the neural network predictions.

    This might repeat some code from earlier.
    """
    delta_x = acts[:,0]
    delta_y = acts[:,1]
    delta_z = acts[:,2]
    str_x = f'$\Delta$x, ({np.min(delta_x):0.4f}, {np.max(delta_x):0.4f})'
    str_y = f'$\Delta$y, ({np.min(delta_y):0.4f}, {np.max(delta_y):0.4f})'
    str_z = f'$\Delta$z, ({np.min(delta_z):0.4f}, {np.max(delta_z):0.4f})'
    str_x += f', shape: {acts.shape}'
    l_x = f'x, {np.mean(delta_x):0.4f} +/- {np.std(delta_x):0.4f}'
    l_y = f'y, {np.mean(delta_y):0.4f} +/- {np.std(delta_y):0.4f}'
    l_z = f'z, {np.mean(delta_z):0.4f} +/- {np.std(delta_z):0.4f}'

    # Are we also plotting rotations?
    plot_rots = acts.shape[1] == 6
    if plot_rots:
        nrows, ncols = 2, 3
        drot_x = acts[:,3]
        drot_y = acts[:,4]
        drot_z = acts[:,5]
        str_drot_x = f'$\Delta$rotx, ({np.min(drot_x):0.4f}, {np.max(drot_x):0.4f})'
        str_drot_y = f'$\Delta$roty, ({np.min(drot_y):0.4f}, {np.max(drot_y):0.4f})'
        str_drot_z = f'$\Delta$rotz, ({np.min(drot_z):0.4f}, {np.max(drot_z):0.4f})'
        l_drot_x = f'x, {np.mean(drot_x):0.4f} +/- {np.std(drot_x):0.4f}'
        l_drot_y = f'y, {np.mean(drot_y):0.4f} +/- {np.std(drot_y):0.4f}'
        l_drot_z = f'z, {np.mean(drot_z):0.4f} +/- {np.std(drot_z):0.4f}'
    else:
        nrows, ncols = 1, 3

    # Scaling the ranges. Original ranges were set assuming that we had divided
    # by 0.004 (or multiplied by 250) so just undo that here.
    factor = 1.
    if not scaling:
        factor = 250

    # Bells and whistles.
    titlesize = 32
    ticksize = 28
    legendsize = 23

    _, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(11*ncols, 7*nrows))
    ax[0,0].set_title(str_x, size=titlesize)
    ax[0,1].set_title(str_y, size=titlesize)
    ax[0,2].set_title(str_z, size=titlesize)
    ax[0,0].hist(delta_x, label=l_x, bins=20, edgecolor='k')
    ax[0,1].hist(delta_y, label=l_y, bins=20, edgecolor='k')
    ax[0,2].hist(delta_z, label=l_z, bins=20, edgecolor='k')
    ax[0,0].set_xlim([-4/factor, 4/factor])
    ax[0,1].set_xlim([-4/factor, 4/factor])
    ax[0,2].set_xlim([-5/factor, 5/factor])
    #ax[0,0].set_yscale('log')
    #ax[0,1].set_yscale('log')
    #ax[0,2].set_yscale('log')

    if plot_rots:
        ax[1,0].set_title(str_drot_x, size=titlesize)
        ax[1,1].set_title(str_drot_y, size=titlesize)
        ax[1,2].set_title(str_drot_z, size=titlesize)
        ax[1,0].hist(drot_x, label=l_drot_x, bins=20, edgecolor='k')
        ax[1,1].hist(drot_y, label=l_drot_y, bins=20, edgecolor='k')
        ax[1,2].hist(drot_z, label=l_drot_z, bins=20, edgecolor='k')
        ax[1,0].set_xlim([-0.1/factor, 0.1/factor])
        ax[1,1].set_xlim([-0.1/factor, 0.1/factor])
        ax[1,2].set_xlim([-0.1/factor, 0.1/factor])
        #ax[1,0].set_yscale('log')
        #ax[1,1].set_yscale('log')
        #ax[1,2].set_yscale('log')

    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    plt.savefig(figname)

    print(f'[replay buffer] See plot: {figname}')
