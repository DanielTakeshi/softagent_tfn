
import tools
import torch
from pytorch3d.transforms import so3_log_map, so3_exponential_map

mse_loss = torch.nn.MSELoss()


def rpmg_forward(in_nd):
    proj_kind = in_nd.shape[1]
    if proj_kind == 6:
        r0 = tools.compute_rotation_matrix_from_ortho6d(in_nd)
    elif proj_kind == 9:
        r0 = tools.symmetric_orthogonalization(in_nd)
    elif proj_kind == 4:
        r0 = tools.compute_rotation_matrix_from_quaternion(in_nd)
    elif proj_kind == 10:
        r0 = tools.compute_rotation_matrix_from_10d(in_nd)
    else:
        raise NotImplementedError
    # return r0
    return r0.transpose(-1, -2)


def rpmg_inverse(R, proj_kind):
    R = R.transpose(-1, -2)

    if proj_kind == 6:
        x = torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)

    elif proj_kind == 9:
        x = R.reshape(-1, 9)

    elif proj_kind == 4:
        x = tools.compute_quaternions_from_rotation_matrices(R)

    elif proj_kind == 10:
        q = tools.compute_quaternions_from_rotation_matrices(R)
        reg_A = torch.eye(4, device=q.device)[None].repeat(q.shape[0], 1, 1) \
            - torch.bmm(q.unsqueeze(-1), q.unsqueeze(-2))
        x = tools.convert_A_to_Avec(reg_A)

    else:
        raise NotImplementedError

    return x


def rpmg_goal_and_nearest(x, R_goal):
    R_goal = R_goal.transpose(-1, -2)
    proj_kind = x.shape[1]

    if proj_kind == 6:
        x_proj_1 = (R_goal[:, :, 0] * x[:, :3]).sum(dim=1,
                                                    keepdim=True) * R_goal[:, :, 0]
        x_proj_2 = (R_goal[:, :, 0] * x[:, 3:]).sum(dim=1, keepdim=True) * R_goal[:, :, 0] \
            + (R_goal[:, :, 1] * x[:, 3:]).sum(dim=1,
                                               keepdim=True) * R_goal[:, :, 1]
        x_goal = torch.cat([R_goal[:, :, 0], R_goal[:, :, 1]], dim=1)
        x_nearest = torch.cat([x_proj_1, x_proj_2], dim=1)

    elif proj_kind == 9:
        x_goal = R_goal.reshape(-1, 9)
        x_nearest = tools.compute_SVD_nearest_Mnlsew(
            x.reshape(-1, 3, 3), R_goal)

    elif proj_kind == 4:
        q_1 = tools.compute_quaternions_from_rotation_matrices(R_goal)
        q_2 = -q_1
        x_proj = tools.normalize_vector(x)
        x_goal = torch.where(
            (q_1 - x_proj).norm(dim=1, keepdim=True) < (q_2 -
                                                        x_proj).norm(dim=1, keepdim=True),
            q_1, q_2)
        x_nearest = (x * x_goal).sum(dim=1, keepdim=True) * x_goal

    elif proj_kind == 10:
        q_goal = tools.compute_quaternions_from_rotation_matrices(R_goal)
        x_nearest = tools.compute_nearest_10d(x, q_goal)
        reg_A = torch.eye(4, device=q_goal.device)[None].repeat(q_goal.shape[0], 1, 1) \
            - torch.bmm(q_goal.unsqueeze(-1), q_goal.unsqueeze(-2))
        x_goal = tools.convert_A_to_Avec(reg_A)

    return x_nearest, x_goal


def projective_manifold_gradient_loss(
        x0_pred, x1_pred, delta_R10_gt,
        step_size=0.05, lambda_reg=0.01,
        transpose=False):
    if transpose:
        R0_pred = rpmg_forward(x0_pred.detach()).transpose(-1, -2)
        R1_pred = rpmg_forward(x1_pred.detach()).transpose(-1, -2)
    else:
        R0_pred = rpmg_forward(x0_pred.detach())
        R1_pred = rpmg_forward(x1_pred.detach())

    delta_r = so3_log_map(torch.bmm(R0_pred.transpose(-1, -2),
                                    torch.bmm(delta_R10_gt, R1_pred)), eps=0.001)
    R0_goal = torch.bmm(R0_pred, so3_exponential_map(step_size*delta_r))

    x0_nearest, x0_goal = rpmg_goal_and_nearest(x0_pred, R0_goal)
    if(False):
        loss = mse_loss(x0_pred, x0_nearest -
                        lambda_reg*(x0_nearest - x0_goal))
    else:
        loss = mse_loss(x0_pred, x0_nearest) \
            + lambda_reg*mse_loss(x0_pred, x0_goal)
    return loss, delta_r.norm(dim=-1)


def projective_manifold_gradient_loss_absolute(
        x0_pred, x1_pred, R0_gt, R1_gt,
        step_size=0.05, lambda_reg=0.01,
        transpose=False):
    if transpose:
        R0_pred = rpmg_forward(x0_pred.detach()).transpose(-1, -2)
        R1_pred = rpmg_forward(x1_pred.detach()).transpose(-1, -2)
    else:
        R0_pred = rpmg_forward(x0_pred.detach())
        R1_pred = rpmg_forward(x1_pred.detach())

    delta_R10_gt = torch.bmm(R0_gt, R1_gt.transpose(-1, -2))
    delta_r = so3_log_map(torch.bmm(R0_pred.transpose(-1, -2),
                                    torch.bmm(delta_R10_gt, R1_pred)))
    R0_goal = torch.bmm(R0_pred, so3_exponential_map(step_size*delta_r))

    x0_nearest, x0_goal = rpmg_goal_and_nearest(x0_pred, R0_goal)
    if(False):
        loss = mse_loss(x0_pred, x0_nearest -
                        lambda_reg*(x0_nearest - x0_goal))
    else:
        loss = mse_loss(x0_pred, x0_nearest) \
            + lambda_reg*mse_loss(x0_pred, x0_goal)
    return loss, delta_r.norm(dim=-1)
