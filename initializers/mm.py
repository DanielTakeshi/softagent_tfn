import numpy as np

from initializers import InitializerBase

class MoveCloserInitializer(InitializerBase):
    THRESH2 = 75 + 125

    def __init__(self, env, args):
        self.env = env

    def get_action(self, obs, info=None):
        step = self.env.inner_step
        T = self.env.tool_idx
        dx, dy, dz = 0., 0., 0.

        # Position of the tool. Unfortunately adding to tx and tz is a real hack.
        tx = self.env.tool_state[T,0]
        ty = self.env.tool_state[T,1]
        tz = self.env.tool_state[T,2]
        tx += 0.090
        tz += 0.090

        # Position of the item.
        rigid_avg_pos = self.env._get_rigid_pos()
        ix = rigid_avg_pos[0]
        iy = rigid_avg_pos[1]
        iz = rigid_avg_pos[2]

        # Unfortunately distance thresholds have to be tuned carefully.
        dist_xz = np.sqrt((tx-ix)**2 + (tz-iz)**2)

        thresh1 = 75
        thresh2 = thresh1 + 125
        if 0 <= step < 15:
            # Only move if the item is below the sphere.
            if dist_xz < 0.050:
                # Go the _negative_ direction
                dx = -(ix - tx)
                dz = -(iz - tz)
        elif 15 <= step < 66:
            # Actually it seems like faster strangely means less water movement.
            dy = -0.0040  # stronger -dy won't have effect due to action bounds
        elif thresh1 <= step < thresh2:
            # Try to correct for the discrepancy
            if dist_xz > 0.004:
                dx = ix - tx
                dz = iz - tz
        elif thresh2 <= step < 600:
            # Ah, it would actually be hard for us to do another xz correction here, since
            # if it causes collision, we'd just stop the motion. :(
            dy = 0.0040
        else:
            pass

        # Try to normalize (to magnitude 1) then downscale by a tuned amount.
        # Unfortunately these numbers just come from tuning / visualizing.
        action = np.array([dx, dy, dz, 0.])
        if (5 <= step < 15) or (thresh1 <= step < thresh2):
            if np.linalg.norm(action) > 0:
                action = action / np.linalg.norm(action) * 0.0020

        if self.env.action_mode == 'translation':
            action = action[:3]
        else:
            raise NotImplementedError()

        # 'Un-scale' to anticipate effect of future scaling in `NormalizedEnv`
        lb = self.env._wrapped_env.action_space.low
        ub = self.env._wrapped_env.action_space.high
        action = (action - lb) / ((ub - lb) * 0.5) - 1.0

        # Check if we're done
        done = step >= (self.THRESH2 - 1)

        return action, done

class SmartMoveCloserInitializer(InitializerBase):
    THRESH2 = 75 + 125

    def __init__(self, env, args):
        self.env = env
        self.state = 0

    def reset(self):
        self.state = 0

    def get_action(self, obs, info=None):
        step = self.env.inner_step
        T = self.env.tool_idx
        dx, dy, dz = 0., 0., 0.

        # Position of the tool. Unfortunately adding to tx and tz is a real hack.
        tx = self.env.tool_state[T,0]
        ty = self.env.tool_state[T,1]
        tz = self.env.tool_state[T,2]
        tx += 0.090
        tz += 0.090

        # Position of the item.
        rigid_avg_pos = self.env._get_rigid_pos()
        ix = rigid_avg_pos[0]
        iy = rigid_avg_pos[1]
        iz = rigid_avg_pos[2]

        # Unfortunately distance thresholds have to be tuned carefully.
        dist_xz = np.sqrt((tx-ix)**2 + (tz-iz)**2)

        thresh1 = 75
        thresh2 = thresh1 + 125
        # if 0 <= step < 15:
        if self.state == 0:
            # Don't collide!
            if -0.2 >= tx or -0.2 >= tz or 0.2 <= tx or 0.2 <= tz:
                dx = -ix
                dz = -iz
            # Only move if the item is below the sphere.
            elif dist_xz < 0.050:
                # Go the _negative_ direction
                dx = -(ix - tx)
                dz = -(iz - tz)
            else:
                self.state += 1
        # elif 15 <= step < 66:
        elif self.state == 1:
            # Actually it seems like faster strangely means less water movement.
            if ty > 0.08:
                dy = -0.0040  # stronger -dy won't have effect due to action bounds
            else:
                self.state += 1
                self.finish_step = step
        # elif thresh1 <= step < thresh2:
        elif self.state == 2:
            # Try to correct for the discrepancy
            if dist_xz > 0.004 and step < self.finish_step + 125:
                dx = ix - tx
                dz = iz - tz
            else:
                self.state += 1
        # elif thresh2 <= step < 600:
        elif self.state == 3:
            # Ah, it would actually be hard for us to do another xz correction here, since
            # if it causes collision, we'd just stop the motion. :(
            dy = 0.0040
            if iy < ty:
                self.state = 0
        else:
            pass

        # Try to normalize (to magnitude 1) then downscale by a tuned amount.
        # Unfortunately these numbers just come from tuning / visualizing.
        action = np.array([dx, dy, dz, 0.])
        # if (5 <= step < 15) or (thresh1 <= step < thresh2):
        if self.state == 0 or self.state == 2:
            if np.linalg.norm(action) > 0:
                action = action / np.linalg.norm(action) * 0.0020

        if self.env.action_mode == 'translation':
            action = action[:3]
        else:
            raise NotImplementedError()

        # 'Un-scale' to anticipate effect of future scaling in `NormalizedEnv`
        lb = self.env._wrapped_env.action_space.low
        ub = self.env._wrapped_env.action_space.high
        action = (action - lb) / ((ub - lb) * 0.5) - 1.0

        # Check if we're done
        done = step >= (self.THRESH2 - 1)

        return action, done
