"""Balance 68600 candidate: current zero command, retained execution history."""
import numpy as np

class Observations:
    def __init__(self, contract):
        self.contract = contract
        self.indices = np.asarray(contract['sdk29_indices'])
        self.default = np.asarray(contract['default_joint_positions'], dtype=np.float32)
        self.history = np.zeros((6, 76), dtype=np.float32)
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.frames = 0

    def reset(self):
        self.history.fill(0)
        self.previous_action.fill(0)
        self.frames = 0

    def record_action(self, action):
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (12,) or not np.isfinite(values).all():
            raise ValueError('Invalid previous action')
        bound = self.contract['output']['training_clip_actions']
        self.previous_action[:] = np.clip(values, -bound, bound)

    def push(self, state, velocity, height, torso_rpy):
        quat = np.asarray(state.quaternion, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if not .8 < norm < 1.2:
            raise ValueError('Invalid IMU quaternion norm')
        w, x, y, z = quat / norm
        gravity = [2*(w*y-x*z), -2*(y*z+w*x), -(1-2*x*x-2*y*y)]
        spec = self.contract['input']
        scales = spec['observation_scales']
        frame = np.concatenate((
            # This policy was trained with exactly zero velocity and height 0.74.
            spec['fixed_velocity_command'], [spec['fixed_height_command']],
            state.omega * scales['ang_vel'], gravity,
            (state.q[self.indices] - self.default) * scales['dof_pos'],
            state.dq[self.indices] * scales['dof_vel'], self.previous_action,
        )).astype(np.float32)
        if frame.shape != (76,) or not np.isfinite(frame).all():
            raise ValueError('Invalid observation frame')
        self.history[:-1] = self.history[1:]
        bound = spec['clip_observations']
        self.history[-1] = np.clip(frame, -bound, bound)
        self.frames += 1
        return self.history.reshape(1, 456)
