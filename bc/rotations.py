import numpy as np
import torch
from rpmg import tools
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as Rot
np.set_printoptions(precision=8)
#from pytorch3d.transforms import quaternion_to_matrix

# === ROTATION REPRESENTATION CONVERTERS ===

# class Example:
#     def convert_to(self, quaternion, tool_rotation):
#         """
#         Parameters: quaternion - a PyQuaternion representing EXTRINSIC rotation
#           tool_rotation - a PyQuaternion representing the current tool rotation

#         Returns: rotation in a new format, in shape (K,), where K is the number
#         of dimensions of the rotation
#         """
#         pass

#     def convert_from(self, example, tool_rotation):
#         """
#         Parameters: rotation in the format, in shape (K,)
#           tool_rotation - a PyQuaternion representing the current tool rotation

#         Returns: a PyQuaternion representing that rotation
#         """
#         pass

class ExtrinsicToIntrinsic:
    def convert_to(self, quaternion, tool_rotation):
        return tool_rotation.inverse * quaternion * tool_rotation

    def convert_from(self, rotation, tool_rotation):
        return tool_rotation * rotation * tool_rotation.inverse

class Compose:
    def __init__(self, *args):
        self.ops = args

    def convert_to(self, quaternion, tool_rotation):
        for op in self.ops:
            quaternion = op.convert_to(quaternion, tool_rotation)

        return quaternion

    def convert_from(self, rotation, tool_rotation):
        for op in self.ops[::-1]:
            rotation = op.convert_from(rotation, tool_rotation)

        return rotation

class RotationMatrix:
    # NOTE(daniel): this does not depend on the rotation_dim, except for how
    # I need to make rtol and atol different for 9D rotations (why? not sure).

    def __init__(self, rot_dim):
        self.rot_dim = rot_dim
        if self.rot_dim == 9:
            self.rtol = 1e-2
            self.atol = 1e-2
        else:
            self.rtol = 1e-5
            self.atol = 1e-5

    def convert_to(self, quaternion, tool_rotation):
        matrix_3x3 = quaternion.rotation_matrix
        return matrix_3x3.flatten()

    def convert_from(self, rotation, tool_rotation):
        # Use pyquaternion 0.9.9 to get rtol and atol args working.
        matrix_3x3 = rotation.reshape((3,3))
        return Quaternion(matrix=matrix_3x3, rtol=self.rtol, atol=self.atol)

class AxisAngle:
    def convert_to(self, quaternion, tool_rotation):
        axis = quaternion.get_axis(undefined=np.array([0., 0., 1.]))
        dtheta = quaternion.radians

        return axis * dtheta

    def convert_from(self, axis_angle, tool_rotation):
        dtheta = np.linalg.norm(axis_angle)
        if dtheta != 0:
            axis = axis_angle / dtheta
            return Quaternion(axis=axis, angle=dtheta)
        else:
            # identity rotation
            return Quaternion()

class FlowConverter(AxisAngle):
    # Flow needs to process quaternion further in the replay buffer
    def convert_to(self, quaternion, tool_rotation):
        return quaternion

class RPMGFlowConverter(RotationMatrix):
    # RPMG outputs a rotation matrix, not axis angle
    def convert_to(self, quaternion, tool_rotation):
        return quaternion

class NoRPMG:
    def __init__(self, rot_dim):
        self.rot_dim = rot_dim

    def convert_to(self, rot_matrix, tool_rotation):
        rot_matrix = rot_matrix.reshape((3, 3))
        R = torch.from_numpy(rot_matrix).unsqueeze(0)

        if self.rot_dim == 6:
            x = torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)
        elif self.rot_dim == 9:
            x = R.reshape(-1, 9)
        elif self.rot_dim == 4:
            x = tools.compute_quaternions_from_rotation_matrices(R)
        elif self.rot_dim == 10:
            q = tools.compute_quaternions_from_rotation_matrices(R)
            reg_A = torch.eye(4, device=q.device)[None].repeat(q.shape[0], 1, 1) \
                - torch.bmm(q.unsqueeze(-1), q.unsqueeze(-2))
            x = tools.convert_A_to_Avec(reg_A)
        else:
            raise NotImplementedError

        return x.flatten().numpy()

    def convert_from(self, rotation, tool_rotation):
        rotation = torch.from_numpy(rotation).unsqueeze(0)
        if self.rot_dim == 6:
            r0 = tools.compute_rotation_matrix_from_ortho6d(rotation)
        elif self.rot_dim == 9:
            r0 = tools.symmetric_orthogonalization(rotation)
        elif self.rot_dim == 4:
            r0 = tools.compute_rotation_matrix_from_quaternion(rotation)
        elif self.rot_dim == 10:
            r0 = tools.compute_rotation_matrix_from_10d(rotation)
        else:
            raise NotImplementedError
        return r0.flatten().numpy()

# === CONVERTER REGISTRATION ===

CONVERTERS = {
    "flow": FlowConverter(),
    "axis_angle": AxisAngle(),
    "intrinsic_axis_angle": Compose(
        ExtrinsicToIntrinsic(),
        AxisAngle(),
    ),
    "rotation_4D": RotationMatrix(rot_dim=4),
    "rotation_6D": RotationMatrix(rot_dim=6),
    "rotation_9D": RotationMatrix(rot_dim=9),
    "rotation_10D": RotationMatrix(rot_dim=10),
    "intrinsic_rotation_6D": Compose(
        ExtrinsicToIntrinsic(),
        RotationMatrix(rot_dim=6),
    ),
    "rpmg_flow_6D": RPMGFlowConverter(rot_dim=6),
    "no_rpmg_6D": Compose(
        RotationMatrix(rot_dim=6),
        NoRPMG(rot_dim=6),
    ),
    "no_rpmg_9D": Compose(
        RotationMatrix(rot_dim=9),
        NoRPMG(rot_dim=9),
    ),
    "no_rpmg_10D": Compose(
        RotationMatrix(rot_dim=10),
        NoRPMG(rot_dim=10),
    ),
}

# === ENV SPECIFIC CANONICALIZATION ===

def _canonicalize_action(env_name, obs_tuple, act_raw, qt_current):
    """
    Gets global delta action (in PyQuaternion form) from raw action and
    observation, depending on environment
    This is especially important as PourWater6D has act_raw as delta euler
    angle, while MixedMedia uses *local* axis-angle. Both need to be
    converted to global axis-angle.
    Note [qt_current] is only used for MixedMedia

    Returns [tool_rotation] the extrinsic rotation of the tool currently
    """
    act_tran = act_raw[:3]
    axis = act_raw[3:]
    dtheta = np.linalg.norm(axis)

    # Unfortunately there are some 'hacks' here. If PourWater, we negate
    # the 3rd value because the axis in SoftGym is actually (0,0,-1) when
    # handling rotations. A positive value means dropping water, and that
    # is clockwise w.r.t. +z, but if we keep it here that would negate it
    # for PyQuaternion. OK to do here as the purpose is to get accurate
    # flow estimates, but check if we can simplify the env as well.
    if env_name == 'PourWater':
        tool_origin = obs_tuple[0][0,:3]  # shape (10,14)
        if dtheta == 0:
            axis = np.array([0., 0., -1.])
        else:
            # FYI, I tried visualizing with and without this, and we do
            # need to negate to see proper flow vectors.
            axis[2] = -axis[2]

        axis = axis / np.linalg.norm(axis)
        delta_quat = Quaternion(axis=axis, angle=dtheta)
        tool_raw = obs_tuple[0][0, 6:10]
        tool_rotation = Quaternion(w=tool_raw[3], x=tool_raw[0], y=tool_raw[1], z=tool_raw[2])
    elif env_name == 'PourWater6D':
        tool_origin = obs_tuple[0][0, :3] # shape (10, 14)

        # Get current global rotation in intrinsic ZYX euler angles due to the way
        # we handle rotation
        curr_rot = Rot.from_quat(obs_tuple[0][0, 6:10])
        curr_euler = curr_rot.as_euler('zyx')
        curr_z, curr_y, curr_x = -curr_euler[0], curr_euler[1], curr_euler[2]

        # Compute new tool orientation
        theta_x = curr_x + act_raw[3]
        theta_y = curr_y + act_raw[4]
        theta_z = curr_z + act_raw[5]
        axis_ang_z = np.array([0., 0., -1.])
        axis_ang_y = np.array([0., 1., 0.])
        axis_ang_x = np.array([1, 0., 0.])
        axis_angle_z = axis_ang_z * theta_z
        axis_angle_y = axis_ang_y * theta_y
        axis_angle_x = axis_ang_x * theta_x
        final_rot = Rot.from_rotvec(axis_angle_x) * Rot.from_rotvec(axis_angle_y) * Rot.from_rotvec(axis_angle_z)

        # Compute rotation difference
        delta_rot = final_rot * curr_rot.inv()
        delta_raw = delta_rot.as_quat()
        delta_quat = Quaternion(w=delta_raw[3], x=delta_raw[0], y=delta_raw[1], z=delta_raw[2])

        # Compute current tool rotation for later if we want to convert to intrinsic rotations
        tool_raw = obs_tuple[0][0, 6:10]
        tool_rotation = Quaternion(w=tool_raw[3], x=tool_raw[0], y=tool_raw[1], z=tool_raw[2])
    elif env_name in ['MMOneSphere', 'MMMultiSphere']:
        tool_origin = obs_tuple[0][:3]  # tool tip position
        if dtheta == 0:
            axis = np.array([0., -1., 0.])

        axis = axis / np.linalg.norm(axis)

        # Change axis from local frame to world to work with 6DoF
        axis_world = qt_current.rotate(axis)
        delta_quat = Quaternion(axis=axis_world, angle=dtheta)

        tool_rotation = qt_current

    return act_tran, tool_origin, delta_quat, tool_rotation

def _decanonicalize_action(env_name, delta_quat, env):
    dtheta = delta_quat.radians

    if (env_name in ['MMOneSphere', 'MMMultiSphere']):
        axis = delta_quat.get_axis(undefined=np.array([0., 0., 1.]))
        if dtheta != 0:
            # Get global rotation axis from env
            curr_q = env.tool_state[0, 6:10]
            qt_current = Quaternion(w=curr_q[3], x=curr_q[0], y=curr_q[1], z=curr_q[2])
            inv_qt_current = qt_current.inverse
            local_axis = inv_qt_current.rotate(axis) * dtheta
            return local_axis
        else:
            return np.zeros(3)
    elif (env_name == 'PourWater'):
        # Now positive rotation is w.r.t., the negative z axis, not the
        # positive z axis. :( Also only do this if we used pytorch geometric.
        # I think this condition should be sufficient but DOUBLE CHECK.
        # UPDATE(06/02/2022), ah also needs to be done with pointwise losses!
        # UPDATE(06/03/2022), if SVD but with MSE after it, don't do this.
        axis = delta_quat.get_axis(undefined=np.array([0., 0., 1.]))
        action = axis * dtheta
        action[2] = -action[2]
        return action
    elif (env_name in ['PourWater6D']):
        # Sadly, we need to convert from delta axis angle to delta euler angle. oof
        if dtheta != 0:
            # Convert from delta axis angle -> delta scipy rotation
            delta_quat._normalise() # knock on wood
            delta_items = delta_quat.elements
            delta_raw = np.array([delta_items[1], delta_items[2], delta_items[3], delta_items[0]])
            delta_rot = Rot.from_quat(delta_raw)

            # Get current env scipy rotation
            curr_rot = Rot.from_quat(env.glass_states[0, 6:10])

            # Perform rotation
            final_rot = delta_rot * curr_rot

            # Convert from final scipy rotation -> final euler angles
            final_euler = final_rot.as_euler('zyx')
            # This flipping of [-final_euler[0]] is where we handle the
            # flipped z-axis in PourWater
            final_z, final_y, final_x = -final_euler[0], final_euler[1], final_euler[2]

            # Convert from final -> delta euler angles
            dtheta_x = final_x - env.glass_rotation_x
            dtheta_y = final_y - env.glass_rotation_y
            dtheta_z = final_z - env.glass_rotation_z

            return np.array([dtheta_x, dtheta_y, dtheta_z])
        else:
            return np.zeros(3)

def _env_to_tool_rotation(env_name, env):
    if (env_name in ['MMOneSphere', 'MMMultiSphere']):
        curr_q = env.tool_state[0, 6:10]
        return Quaternion(w=curr_q[3], x=curr_q[0], y=curr_q[1], z=curr_q[2])
    elif env_name in ['PourWater', 'PourWater6D']:
        curr_q = env.glass_states[0, 6:10]
        return Quaternion(w=curr_q[3], x=curr_q[0], y=curr_q[1], z=curr_q[2])

# === DRIVER CODE ===

def convert_action(rotation_representation, env_name, obs_tuple, act_raw, qt_current):
    assert rotation_representation in CONVERTERS, f"Invalid rotation representation {rotation_representation}"
    act_tran, tool_origin, delta_quat, tool_rotation = _canonicalize_action(env_name, obs_tuple, act_raw, qt_current)
    converted_rotation = CONVERTERS[rotation_representation].convert_to(delta_quat, tool_rotation)
    return act_tran, tool_origin, converted_rotation

def deconvert_action(rotation_representation, env_name, action, env):
    assert rotation_representation in CONVERTERS, f"Invalid rotation representation {rotation_representation}"
    tool_rotation = _env_to_tool_rotation(env_name, env)
    delta_quat = CONVERTERS[rotation_representation].convert_from(action[3:], tool_rotation)
    env_specific_rotation = _decanonicalize_action(env_name, delta_quat, env)
    return np.concatenate((action[:3], env_specific_rotation))
