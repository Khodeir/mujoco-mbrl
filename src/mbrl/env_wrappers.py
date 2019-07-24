from dm_control import suite
from dm_control.rl import environment
import numpy as np


class EnvWrapper(environment.Base):
    def __init__(self, env):
        action_spec = env.action_spec()
        self._minimum = action_spec.minimum
        self._maximum = action_spec.maximum
        self._env = env

    @staticmethod
    def load(env_name, task_name, **kwargs):
        env = suite.load(env, task_name, **kwargs)
        wrapper_classname = "".join([part.capitalize() for part in env_name.split("_")])
        try:
            wrapper = eval(wrapper_classname)(env)
            return wrapper
        except NameError:
            raise NameError("No wrapper for {}".format(env_name))

    def get_state(self):
        # TODO: Do something about the fact that get_state has a potentially
        # different dimension than sample_state
        return self._env.physics.state()

    def sample_state(self):
        raise NotImplementedError

    def sample_action(self):
        minimum = max(
            self.action_spec.minimum[0], -3
        )  # Clipping because LQR task has INF bounds
        maximum = min(self.action_spec.maximum[0], 3)
        action = np.random.uniform(minimum, maximum, self.action_spec.shape[0])
        return action

    def get_goal_weights(self):
        self._state_penalty = 1.0
        weights = np.zeros(self.state_dim)
        return weights


class PointMass(EnvWrapper):
    state_dim = 4

    def get_goal_weights(self):
        weights = super().goal_weights()
        weights[0:2] = 10 * self._state_penalty
        weights[2:] = (
            self._state_penalty / 4.0
        )  # Penalties on the velocities act as dampers
        return weights


class Reacher(EnvWrapper):
    state_dim = 4
    def sample_state(self):
        state_dim = 4
        state = np.zeros(self.state_dim)

        state[0] = np.random.uniform(
            low=-np.pi, high=np.pi
        )  # 2*np.pi*np.random.rand()- np.pi
        state[1] = np.random.uniform(low=-2.8, high=2.8)  # Avoid infeasible goals
        state[2] = np.random.uniform(low=-3, high=3)
        state[3] = np.random.uniform(low=-3, high=3)

        return state

    def get_goal_weights(self):
        weights = super().goal_weights()

        weights[0:2] = self._state_penalty

        weights[2:] = (
            self._state_penalty / 20.0
        )  # Penalties on the velocities act as dampers
        return weights


class Cheetah(EnvWrapper):
    state_dim = 18 - 1 + 2

    def sample_state(self):
        state_dim = 18
        state = np.zeros(state_dim)
        state[1] = np.random.uniform(-0.2, 0.2)  # Vertical Height
        if state[1] > 0.05:
            state[2] = np.random.uniform(-3.14, 3.14)  # torso angle (0, 50)
        else:
            if np.random.uniform() < 0.72:
                state[2] = np.random.uniform(-3.14, -1.5)  # torso angle (-50, 0)
            else:
                state[2] = np.random.uniform(2.5, 3.14)
        state[3] = np.random.uniform(
            -0.5236, 1.0472
        )  # bthigh (-30, 60) = (-0.5236, 1.0472)
        state[4] = np.random.uniform(
            -0.8727, 0.8727
        )  # bshin (-50, 50) = (-0.8727, 0.8727)
        state[5] = np.random.uniform(
            -4.0143, 0.8727
        )  # bfoot (-230, 50) = (-4.0143, 0.8727)
        state[6] = np.random.uniform(
            -0.9948, 0.0070
        )  # fthigh (-57, .4) = (-0.9948, 0.0070)
        state[7] = np.random.uniform(
            -1.2217, 0.8727
        )  # fshin (-70, 50) = (-1.2217, 0.8727)
        state[8] = np.random.uniform(
            -0.4887, 0.4887
        )  # ffoot (-28, 28) = (-0.4887, 0.4887)
        state[9:] = np.random.uniform(-3, 3, 9)  # Velocities

        return state

    def get_state(self):
        state = super().get_state()[1:]
        state = np.append(state, self._env.physics.speed())  # Add horizontal speed
        state = np.append(
            state, self._env.physics.named.data.subtree_com["torso"][2]
        )  # Add torso height
        return state

    def get_goal_weights(self):
        weights = super().goal_weights()
        weights[17] = self._state_penalty
        weights[18] = self._state_penalty / 2.0
        return weights


class Manipulator(EnvWrapper):
    state_dim = 22 + 7

    def get_state(self):
        state = super().get_state()
        state = np.append(
            state, self._env.physics.named.data.site_xpos["grasp", "x"]
        )  # gripper position
        state = np.append(
            state, self._env.physics.named.data.site_xpos["grasp", "z"]
        )  # gripper position
        state = np.append(state, self._env.physics.touch())  # Sensors. 5 dimensions
        return state

    def get_goal_weights(self):
        weights = super().goal_weights()
        weights[8:10] = 10 * self._state_penalty
        weights[10:21] = self._state_penalty / 4
        weights[-7:-5] = 10 * self._state_penalty
        weights[-5:] = self._state_penalty / 20
        return weights


class Humanoid(EnvWrapper):
    state_dim = 55 + 5

    def sample_state(self):
        state_dim = 55
        state = np.zeros(state_dim)  # Removed 0,1 and 3,4,5,6

        state[2] = 1.3  # np.random.uniform(1.45, 1.55) ## Vertical Position
        state[7] = np.random.uniform(
            -0.7854, 0.7854
        )  # abdomen_z (-45, 45) = (-0.7854, 0.7854)
        state[8] = np.random.uniform(
            -1.3089, 0.5236
        )  # abdomen_y (-75, 30) = (-1.3089, 0.5236)
        state[9] = np.random.uniform(
            -0.6109, 0.6109
        )  # abdomen_x (-35, 35) = (-0.6109, 0.6109)

        state[10] = np.random.uniform(
            -0.4363, 0.0873
        )  # right_hip_x (-25, 5) = (-0.4363, 0.0873)
        state[11] = np.random.uniform(
            -1.0472, 0.6109
        )  # right_hip_z (-60, 35) = (-1.0472, 0.6109)
        state[12] = np.random.uniform(
            -1.9199, 0.3491
        )  # right_hip_y (-110, 20) = (-1.9199, 0.3491)
        state[13] = np.random.uniform(
            -2.7925, 0.0349
        )  # right_knee (-160, 2) = (-2.7925, 0.0349)
        state[14] = np.random.uniform(
            -0.8727, 0.8727
        )  # right_ankle_y (-50, 50) = (-0.8727, 0.8727)
        state[15] = np.random.uniform(
            -0.8727, 0.8727
        )  # right_ankle_x (-50, 50) = (-0.8727, 0.8727)

        state[16] = np.random.uniform(
            -0.4363, 0.0873
        )  # left_hip_x (-25, 5) = (-0.4363, 0.0873)
        state[17] = np.random.uniform(
            -1.0472, 0.6109
        )  # left_hip_z (-60, 35) = (-1.0472, 0.6109)
        state[18] = np.random.uniform(
            -1.9199, 0.3491
        )  # left_hip_y (-110, 20) = (-1.9199, 0.3491)
        state[19] = np.random.uniform(
            -2.7925, 0.0349
        )  # left_knee (-160, 2) = (-2.7925, 0.0349)
        state[20] = np.random.uniform(
            -0.8727, 0.8727
        )  # left_ankle_y (-50, 50) = (-0.8727, 0.8727)
        state[21] = np.random.uniform(
            -0.8727, 0.8727
        )  # left_ankle_x (-50, 50) = (-0.8727, 0.8727)

        state[22] = np.random.uniform(
            -1.4835, 1.0472
        )  # right_shoulder1 (-85, 60) = (-1.4835, 1.0472)
        state[23] = np.random.uniform(
            -1.4835, 1.0472
        )  # right_shoulder2 (-85, 60) = (-1.4835, 1.0472)
        state[24] = np.random.uniform(
            -1.5708, 0.8727
        )  # right_elbow (-90, 50) = (-1.5708, 0.8727)

        state[25] = np.random.uniform(
            -1.0472, 1.4835
        )  # left_shoulder1 (-60, 85) = (-1.0472, 1.4835)
        state[26] = np.random.uniform(
            -1.0472, 1.4835
        )  # left_shoulder2 (-60, 85) = (-1.0472, 1.4835)
        state[27] = np.random.uniform(
            -1.5708, 0.8727
        )  # left_elbow (-90, 50) = (-1.5708, 0.8727)

        # state[34:] = 0.1*np.random.uniform(-0.1, 0.1, 21)  # Velocities

        return state

    def sample_action(self):
        action = np.random.normal(0, 0.4, self.action_dim)
        action[3:-6] = 0.0
        return action

    def get_state(self):
        state = super().get_state()  # 55 (pure state)
        # state = np.append(state, self.env.physics.head_height()) ## +1 head height (target should be 1.6)
        # state = np.append(state, self.env.physics.torso_upright()) ## +1 torso projected height (target should be 1.0)
        com_pos = self.env.physics.center_of_mass_position()
        rfoot = self.env.physics.named.data.xpos["right_foot"]
        lfoot = self.env.physics.named.data.xpos["left_foot"]
        ave_foot = (rfoot + lfoot) / 2.0
        above_feet = ave_foot + np.array([0.0, 0.0, 1.3])
        torso = self.env.physics.named.data.xpos["torso"]

        first_penalty = np.linalg.norm(
            com_pos[:2] - ave_foot[:2]
        )  # First described by Tassa
        second_penalty = np.linalg.norm(
            com_pos[:2] - torso[:2]
        )  # Second term described by Tassa
        third_penalty = np.linalg.norm(torso[1:] - above_feet[1:])  # Third term

        state = np.append(state, first_penalty)  # +1
        state = np.append(state, second_penalty)  # +1
        state = np.append(state, third_penalty)  # +1
        state = np.append(
            state, self.env.physics.center_of_mass_velocity()[:2]
        )  # +2 velocity should be 0
        return state

    def get_goal_weights(self):
        weights = super().goal_weights()
        weights[-5:] = 10 * self._state_penalty
        return weights


class Swimmer(EnvWrapper):
    state_dim = 10 + 2

    def sample_state(self):
        state_dim = 10
        # Assuming swimmer3
        state = np.zeros(state_dim)

        state[2] = np.random.uniform(low=-3, high=3)

        return state

    def get_state(self):
        state = super().get_state()  # 16-2 dims
        state = np.append(
            state, self.env.physics.named.data.xmat["head"][:2]
        )  # Head orientation (2)
        return state

    def get_goal_weights(self):
        weights = super().goal_weights()
        weights[0:1] = 10 * self._state_penalty  # Distance to target
        # weights[18] = self._state_penalty/2.
        weights[5:-2] = self._state_penalty  # velocities
        # weights[8:-2] = self._state_penalty/4. ## velocities
        return weights


class Walker(EnvWrapper):
    state_dim = 18 - 1 + 3

    def sample_state(self):
        state_dim = 18
        state = np.zeros(state_dim)
        # state[0] = np.random.uniform() ## horizontal position
        # state[1] = 0#np.random.uniform(0, 0.05) ## vertical position
        state[2] = np.random.uniform(-0.1, 0.1)  # clock rotation of main body

        hip_rot = np.random.uniform(-0.15, 0.15)
        state[3] = hip_rot  # right_hip (-20, 100)   = (-0.3491, 1.7452)
        state[4] = np.random.uniform(-0.3, 0)  # right_knee (-150, 0)   = (-2.6178 , 0)
        state[5] = np.random.uniform(
            -0.1, 0.1
        )  # right_ankle (-45, 45)  = (-0.7854, 0.7854)
        state[
            6
        ] = (
            -hip_rot
        )  # np.random.uniform(-0.3491, 1.) ## left_hip (-20, 100)   = (-0.3491, 1.7452)
        state[7] = np.random.uniform(-0.3, 0)  # left_knee (-150, 0)   = (-2.6178 , 0)
        state[8] = np.random.uniform(
            -0.1, 0.1
        )  # left_ankle (-45, 45)  = (-0.7854, 0.7854)

        # state[9:] = np.random.uniform(-0.04, 0.04, 9) ## Velocities

        return state

    def get_state(self):
        state = super().get_state()[1:]
        state = np.append(state, self.env.physics.torso_upright())  # Add torso upright
        state = np.append(state, self.env.physics.torso_height())  # Add torso height
        state = np.append(
            state, self.env.physics.horizontal_velocity()
        )  # Add horizontal speed
        return state

    def get_goal_weights(self):
        weights = super().goal_weights()
        weights[-3:] = self._state_penalty
        return weights


class Hopper(EnvWrapper):
    state_dim = 14 - 1 + 4

    def sample_state(self):
        state_dim = 14
        state = np.zeros(state_dim)
        # state[0] = np.random.uniform() ## horizontal position
        state[1] = -0.078789  # np.random.uniform(0, 0.05) ## vertical position
        state[2] = np.random.uniform(-0.01, 0.01)  # clock rotation of main body
        state[3] = np.random.uniform(
            -0.01, 0.01
        )  # clock rotation of main body right_hip (-20, 100) = (-0.3491, 1.7452)
        state[4] = np.random.uniform(-0.01, 0.01)  # hip (-170, 10)   = (-2.6178 , 0)
        state[5] = np.random.uniform(0.1, 0.12)  # knee (5, 150)  = (-0.7854, 0.7854)
        state[6] = np.random.uniform(
            -0.01, 0.01
        )  # ankle (-45, 45)  = (-0.7854, 0.7854)

        state[7:] = np.random.uniform(-0.01, 0.01, 7)  # Velocities

        return state

    def get_state(self):
        state = super().get_state()[1:]  # Removed horizontal position
        state = np.append(state, self.env.physics.touch())  # Add touch sensors in feet
        state = np.append(state, self.env.physics.height())  # Add torso height
        state = np.append(state, self.env.physics.speed())  # Add horizontal speed
        return state

    def get_goal_weights(self):
        weights = super().goal_weights()
        weights[-2] = self._state_penalty / 2.0
        weights[-1] = self._state_penalty
        return weights
