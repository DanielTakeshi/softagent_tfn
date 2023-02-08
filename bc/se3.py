"""From Brian Okorn and Chuer Pan.
https://github.com/r-pad/equivariant_pose_graph/tree/main/python/equivariant_pose_graph/utils

We're probably inerested in `flow2pose()`.
"""
import os
import sys
import pickle
from pytorch3d.transforms import (
    Transform3d, Rotate, rotation_6d_to_matrix, axis_angle_to_matrix,
    matrix_to_quaternion, matrix_to_euler_angles, quaternion_to_axis_angle
)
import torch
import numpy as np


def to_transform3d(x, rot_function = rotation_6d_to_matrix):
    trans = x[:,:3]
    rot = x[:,3:]
    return Transform3d(device=x.device).compose(Rotate(rot_function(rot),
        device=rot.device)).translate(trans)


def transform_points(points, x, rot_function = rotation_6d_to_matrix):
    t = x[:,:3]
    R = rot_function(x[:,3:])
    return (torch.bmm(R, points.transpose(-2,-1)) + t.unsqueeze(-1)).transpose(-2,-1)


def transform3d_to(T, device):
    T = T.to(device)
    T = T.to(device)
    T._transforms = [
        t.to(device) for t in T._transforms
    ]
    return T


# def random_se3(N, rot_var = np.pi/180 * 5, trans_var = 0.1, device = None):
#     T =  se3_exp_map(torch.cat(
#         [
#             torch.randn(N,3, device=device)*trans_var,
#             torch.randn(N,3, device=device)*rot_var,
#         ], dim=-1))
#     return Transform3d(matrix=T, device=device)


def random_se3(N, rot_var = np.pi/180 * 5, trans_var = 0.1, device = None):
    R = axis_angle_to_matrix(torch.randn(N,3, device=device)*rot_var)
    t = torch.randn(N,3, device=device)*trans_var
    return Rotate(R, device=device).translate(t)


def symmetric_orthogonalization(M):
    """Maps arbitrary input matrices onto SO(3) via symmetric orthogonalization.
    (modified from https://github.com/amakadia/svd_for_pose)

    M: should have size [batch_size, 3, 3]

    Output has size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    U, _, Vh = torch.linalg.svd(M)
    det = torch.det(torch.bmm(U, Vh)).view(-1, 1, 1)
    Vh = torch.cat((Vh[:, :2, :], Vh[:, -1:, :] * det), 1)
    R = U @ Vh
    return R


def flow2pose(xyz, flow, weights=None, return_transform3d=False,
        return_quaternions=False, world_frameify=True):
    """Flow2Pose via SVD.

    Operates on minibatches of `B` point clouds, each with `N` points. Assumes
    all point clouds have `N` points, but in practice we only call this with
    minibatch size 1 and we get rid of non-tool points before calling this.

    Parameters
    ----------
    xyz: point clouds of shape (B,N,3). This gets zero-centered so it's OK if it
        is not already centered.
    flow: corresponding flow of shape (B,N,3). As with xyz, it gets zero-centered.
    weights: weights for the N points, set to None for uniform weighting, for now
        I don't think we want to weigh certain points more than others, and it
        could be tricky when points can technically be any order in a PCL.
    return_transform3d: Used if we want to return a transform, for which we apply
        on a set of point clouds. This is what Brian/Chuer use to compute losses,
        by applying this on original points and comparing point-wise MSEs.
    return_quaternions: Use if we want to convert rotation matrices to quaternions.
        Uses format of (wxyz) format, so the identity quanternion is (1,0,0,0).
    """
    if weights is None:
        weights = torch.ones(xyz.shape[:-1], device=xyz.device)
    ww = (weights / weights.sum(dim=-1, keepdims=True)).unsqueeze(-1)

    # xyz_mean shape: ((B,N,1), (B,N,3)) mult -> (B,N,3) -> sum -> (B,1,3)
    xyz_mean = (ww * xyz).sum(dim=1, keepdims=True)
    xyz_demean = xyz - xyz_mean  # broadcast `xyz_mean`, still shape (B,N,3)

    # As with xyz positions, find (weighted) mean of flow, shape (B,1,3).
    flow_mean = (ww * flow).sum(dim=1, keepdims=True)

    # Zero-mean positions plus zero-mean flow to find new points.
    xyz_trans = xyz_demean + flow - flow_mean  # (B,N,3)

    # Batch matrix-multiply, get X: (B,3,3), each (3x3) matrix is in SO(3).
    X = torch.bmm(xyz_demean.transpose(-2,-1),  # (B,3,N)
                  ww * xyz_trans)               # (B,N,3)

    # Rotation matrix in SO(3) for each mb item, (B,3,3).
    R = symmetric_orthogonalization(X)

    # 3D translation vector for eacb mb item, (B,3) due to squeezing.
    if world_frameify:
        t = (flow_mean + xyz_mean - torch.bmm(xyz_mean, R)).squeeze(1)
    else:
        t = flow_mean.squeeze(1)

    if return_transform3d:
        return Rotate(R).translate(t)
    if return_quaternions:
        quats = matrix_to_quaternion(matrix=R)
        return quats, t
    return R, t

eps = 1e-9


def dualflow2pose(xyz, flow, polarity, weights = None, return_transform3d = False):
    if(weights is None):
        weights = torch.ones(xyz.shape[:-1], device=xyz.device)
    w = (weights / weights.sum(dim=-1, keepdims=True)).unsqueeze(-1)
    w_p = (polarity * weights).unsqueeze(-1)
    w_p_sum = w_p.sum(dim=1, keepdims=True)
    w_p = w_p / w_p_sum.clamp(min=eps)
    w_n = ((1-polarity) * weights).unsqueeze(-1)
    w_n_sum = w_n.sum(dim=1, keepdims=True)
    w_n = w_n / w_n_sum.clamp(min=eps)


    xyz_mean_p = (w_p * xyz).sum(dim=1, keepdims=True)
    xyz_demean_p = xyz - xyz_mean_p
    xyz_mean_n = (w_n * xyz).sum(dim=1, keepdims=True)
    xyz_demean_n = xyz - xyz_mean_n

    flow_mean_p = (w_p * flow).sum(dim=1, keepdims=True)
    flow_demean_p = flow - flow_mean_p
    flow_mean_n = (w_n * flow).sum(dim=1, keepdims=True)
    flow_demean_n = flow - flow_mean_n

    mask = (polarity.unsqueeze(-1).expand(-1,-1,3)==1)
    xyz_1 = torch.where(mask,
        xyz_demean_p, xyz_demean_n + flow_demean_n)
    xyz_2 = torch.where(mask,
        xyz_demean_p + flow_demean_p, xyz_demean_n)

    X = torch.bmm(xyz_1.transpose(-2,-1), w*xyz_2)

    R = symmetric_orthogonalization(X)
    t_p = (flow_mean_p + xyz_mean_p - torch.bmm(xyz_mean_p, R))
    t_n = (xyz_mean_n - torch.bmm(flow_mean_n + xyz_mean_n, R))

    t = ((w_p_sum * t_p + w_n_sum * t_n)/(w_p_sum + w_n_sum)).squeeze(1)

    if(return_transform3d):
        return Rotate(R).translate(t)
    return R, t


def points2pose(xyz1, xyz2, weights = None, return_transform3d = False):
    if(weights is None):
        weights = torch.ones(xyz1.shape[:-1], device=xyz1.device)
    w = (weights / weights.sum(dim=-1, keepdims=True)).unsqueeze(-1)

    xyz1_mean = (w * xyz1).sum(dim=1, keepdims=True)
    xyz1_demean = xyz1 - xyz1_mean

    xyz2_mean = (w * xyz2).sum(dim=1, keepdims=True)
    xyz2_demean = xyz2 - xyz2_mean

    X = torch.bmm(xyz1_demean.transpose(-2,-1),
                  w*xyz2_demean)

    R = symmetric_orthogonalization(X)
    t = (xyz2_mean - torch.bmm(xyz1_mean, R)).squeeze(1)

    if(return_transform3d):
        return Rotate(R).translate(t)
    return R, t


def _debug_bc_data(device):
    """Debug our Behavioral Cloning data.

    Particularly, the data with the extremely simple 1DoF rotation about the
    y axis. Given the ground-truth flow, we should get the desired rotation.
    If this doesn't work, there's a problem with SVD. If it does work, then
    we at least know this step is OK?

    This could also be useful for debugging the translation corrections we
    use, since the center of rotation is not the centroid of the tool, but
    at the tip of the tool's stick.
    """
    def get_obs_tool_flow(pcl, tool_flow):
        # Copied from bc utils
        pcl_tool = pcl[:,3] == 1
        tf_pts = tool_flow['points']
        tf_flow = tool_flow['flow']
        n_tool_pts_obs = np.sum(pcl_tool)
        n_tool_pts_flow = tf_pts.shape[0]
        # First shapes only equal if: (a) fewer than max pts or (b) no item/distr.
        assert tf_pts.shape[0] <= pcl.shape[0], f'{tf_pts.shape}, {pcl.shape}'
        assert tf_pts.shape == tf_flow.shape, f'{tf_pts.shape}, {tf_flow.shape}'
        assert n_tool_pts_obs == n_tool_pts_flow, f'{n_tool_pts_obs}, {n_tool_pts_flow}'
        assert np.array_equal(pcl[:n_tool_pts_obs,:3], tf_pts)  # yay :)
        a = np.zeros((pcl.shape[0], 3))  # all non-tool point rows get 0s
        a[:n_tool_pts_obs] = tf_flow   # actually encode flow for BC purposes
        return (pcl, a)

    # This is the v06 with the 1DoF rotation about y axis at the start.
    PATH = os.path.join(
        '/data/seita/softgym_mm/data_demo/',
        'MMOneSphere_v01_BClone_filtered_ladle_algorithmic_v06_nVars_2000_obs_combo_act_translation_axis_angle',
        'BC_0000_600.pkl'
    )
    with open(PATH, 'rb') as fh:
        data = pickle.load(fh)

    # obses: list of tuples (different obs types), acts: list of axis-angles.
    obses = data['obs']
    acts = data['act_raw']
    assert len(obses) == len(acts)

    # Remember that to get flow we need the _next_ observation.
    t = 0
    obs_t = obses[t][3]  # PCL at idx=3
    act_t = acts[t]  # the axis-angle formulation
    info_tp1 = obses[t+1][4]  # Flow at idx=4 for _next_ time step.
    obs_t, act_flow_t = get_obs_tool_flow(obs_t, info_tp1)
    print(f'Testing PCLs at t={t} shaped {obs_t.shape} with action: {act_t}')
    print(f'  act_flow_t: {act_flow_t.shape}')

    # Let's visualize this, but make flow longer for clarity. EDIT: for some
    # reason I can't create it here, so let's put it in my MWE.

    # Get the pose from the flow.
    xyz = torch.as_tensor(np.array([obs_t[:,:3]])).float()
    flow = torch.as_tensor(np.array([act_flow_t])).float()
    print(f'Calling flow2pose, xyz: {xyz.shape}, flow: {flow.shape}')
    pose = flow2pose(
        xyz=xyz.to(device),
        flow=flow.to(device),
        weights=None,
        return_transform3d=False,
        return_quaternions=False,
    )

    # Debugging, could be useful to show a visualization?
    R, t = pose
    print(f'\nFinished SVD! Shape of R, t: {R.shape}, {t.shape}\n')
    R = R[0]
    t = t[0]
    print('\nThe rotation and translation:')
    print(R)
    print(t)
    print('\nIs this a rotation matrix? This should be the identity.')
    RT_times_R = torch.matmul(torch.transpose(R, 0, 1), R)
    print(RT_times_R)
    print('\nEuler angles:')
    print(matrix_to_euler_angles(R, convention='XYZ'))
    print('\nQuaternion:')
    quat = matrix_to_quaternion(R)
    print(quat)
    print('\nAxis-angle:')
    # WAIT! This is producing the OPPOSITE rotation!
    print(quaternion_to_axis_angle(quat))
    sys.exit()


if __name__ == "__main__":
    # Try debugging / testing the flow / SVD methods.
    np.set_printoptions(suppress=True, precision=6, linewidth=150, edgeitems=20)
    torch.set_printoptions(sci_mode=False, precision=6, linewidth=150, edgeitems=20)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #_debug_bc_data(device)

    # Minibatch of B point clouds, 10 points in it, with 3D position or flow.
    B = 4
    xyz = torch.rand(B,100,3)
    flow = torch.rand(B,100,3)
    ret_trans = False
    ret_quat = False
    world_frameify = True

    ## For debugging, let's actually do translation only for one item.
    ## If we do this, we get a tensor of (1,0,0,0) as output quaternion (good).
    #flow[0] = torch.ones((1,3)) * 0.5
    #flow[0,0,:] = torch.ones((1,3))

    # More debugging: what if we just scale the first item (pos and flow)?
    # If world_frameify=False, translations are the same subject to positioning
    # of decimal points (good).
    # Rotations are _almost_ the same but up to 3 decimal points:
    #   axis-angle: tensor([-0.091616, -0.033090,  0.173777]), magn: 0.1502
    #   axis-angle: tensor([-0.091667, -0.033330,  0.174000]), magn: 0.1503
    #   axis-angle: tensor([-0.091644, -0.033659,  0.173811]), magn: 0.1506
    #   axis-angle: tensor([-0.091447, -0.033111,  0.173603]), magn: 0.1504
    # It does seem to be close enough and might not matter too much in practice.
    # Magntiudes seem to be close enough that the policy will do similar things.
    #
    # What if world_frameify=True? That would only affect the quality of the
    # resulting returned translation. It still scales well though not as perfectly
    # as just averaging the flow vectors (makes sense).
    xyz[1] = xyz[0] * 10.
    flow[1] = flow[0] * 10.
    xyz[2] = xyz[0] * 100.
    flow[2] = flow[0] * 100.
    xyz[3] = xyz[0] * 1000.
    flow[3] = flow[0] * 1000.

    # Get the pose from the flow.
    pose = flow2pose(
        xyz=xyz.to(device),
        flow=flow.to(device),
        weights=None,
        return_transform3d=ret_trans,  # Brian/Chuer use True
        return_quaternions=ret_quat,   # something new
        world_frameify=world_frameify, # something new
    )
    if ret_trans:
        sys.exit()

    # Debugging, could be useful to show a visualization?
    R, t = pose
    print(f'Shape of R, t: {R.shape}, {t.shape}\n')
    for b in range(B):
        print(f'\n---------- On minibatch item {b} ----------')
        print(R[b])
        quat_b = matrix_to_quaternion(R[b])
        aang_b = quaternion_to_axis_angle(quat_b)
        print(f'quaternion: {quat_b}')
        print(f'axis-angle: {aang_b} with magnitude: {torch.norm(aang_b):0.4f}')
        print(f'translation: {t[b]}')
        if not ret_quat:
            print(f'To confirm rotation matrix (also check it is on cuda):')
            RTR = torch.matmul(torch.transpose(R[b], 0, 1), R[b])
            print(RTR)
