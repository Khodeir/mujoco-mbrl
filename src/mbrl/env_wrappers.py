from typing import Tuple, Dict, Callable, Optional
import torch
import numpy as np
from dm_control import suite
import dm_env
from src.mbrl.data import Rollout
from src.mbrl.utils import Recorder
# TODO: Need to make set_goal consistent with observation_dim
# or at least have parallel implementations that are.
class EnvWrapper(dm_env.Environment):
    def __init__(self, env, flat_obs, env_name, task_name):
        self._env = env
        self._state_penalty = 1.0
        self.action_dim = env.action_spec().shape[0]
        self._action_spec = env.action_spec()
        self._flat_obs = flat_obs
        self._env_name = env_name
        self._task_name = task_name

    @staticmethod
    def load(env_name, task_name, flat_obs=True, **kwargs):
        wrapper_classname = "".join([part.capitalize() for part in env_name.split("_")])
        try:
            wrapper_class = eval(wrapper_classname)
            environment_kwargs = kwargs['environment_kwargs'] = kwargs.get('environment_kwargs', {})
            environment_kwargs["flat_observation"] = flat_obs
            if hasattr(wrapper_class, 'override_control_timestep'):
                print('Overriding control time step')
                environment_kwargs['control_timestep'] = wrapper_class.override_control_timestep
            env = suite.load(env_name, task_name, **kwargs)
            wrapper = eval(wrapper_classname)(env, flat_obs=flat_obs, env_name=env_name, task_name=task_name)
            return wrapper
        except NameError:
            raise NameError("No wrapper for {}".format(env_name))

    def get_state(self) -> torch.Tensor:
        # TODO: Do something about the fact that get_state has a potentially
        # different dimension than sample_state
        return torch.tensor(self._env.physics.state(), dtype=torch.float32)

    def sample_state(self) -> torch.Tensor:
        raise NotImplementedError

    def set_goal(self) -> torch.Tensor:
        raise NotImplementedError

    def sample_action(self, batch_size=None) -> torch.Tensor:
        return self._sample_action(self.action_spec(), batch_size)

    @staticmethod
    def _sample_action(action_spec, batch_size=None) -> torch.Tensor:
        minimum = max(
            action_spec.minimum[0], -3
        )  # Clipping because LQR task has INF bounds
        maximum = min(action_spec.maximum[0], 3)
        if batch_size is None:
            action = np.random.uniform(minimum, maximum, action_spec.shape[0])
        else:
            action = np.random.uniform(
                minimum, maximum, size=action_spec.shape[0] * batch_size
            ).reshape((batch_size, -1))
        return torch.tensor(action, dtype=torch.float32)

    def get_goal_weights(self) -> torch.Tensor:
        weights = torch.zeros(self.state_dim)
        return weights

    def reset(self):
        t = self._env.reset()
        return self._parse_timestep(t)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec
    
    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor], bool]:
        t = self._env.step(np.array(action))
        return self._parse_timestep(t)
       
    def _parse_timestep(self, t) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor], bool]:
        if self._flat_obs:
            obs = torch.tensor(t.observation['observations'], dtype=torch.float32)
        else:
            obs = {
                str(k): torch.tensor(v, dtype=torch.float32)
                for k, v in t.observation.items()
            }
        reward = (
            torch.tensor(t.reward, dtype=torch.float32)
            if t.reward is not None
            else None
        )
        return self.get_state(), obs, reward, t.last()


    def get_rollout(
        self,
        num_steps: int,
        get_action: Callable[[Dict[str, torch.Tensor]], torch.Tensor] = None,
        step_callback: Optional[Callable] = None,
        set_state: bool = False,
        goal_state: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None
    ) -> Rollout:
        if get_action is None:
            get_action = lambda state: self.sample_action()

        states, actions, observations, rewards = [], [], [], []

        state, observation, _, _ = self.reset()
        if set_state:
            initial_state = self.sample_state() if initial_state is None else initial_state
        else:
            # the current state obtained from self.reset
            initial_state = self._env.physics.state()
        # if we need to modify the goal_state, we'll have to also reset the initial state
        if goal_state is not None and hasattr(self, 'set_target'):
            with self._env.physics.reset_context():
                self._env.physics.set_state(initial_state)
                self.set_target(goal_state)
            state, observation, _, _ = self.step(self.sample_action())

        states.append(state)
        observations.append(observation)
        rewards.append(None)
        for timestep in range(num_steps):
            action = get_action(dict(timestep=timestep, state=state, observation=observation))
            actions.append(action)
            state, observation, reward, done = self.step(action)
            states.append(state)
            observations.append(observation)
            rewards.append(reward)
            if step_callback is not None:
                step_callback(timestep)
            if done:
                break

        return Rollout(
            states=states,
            observations=observations,
            actions=actions,
            rewards=rewards[1:],
        )

    def record_rollout(self, *args, mp4path=None, **kwargs):
        with Recorder() as recorder:
            record_frame = lambda t: recorder.record_frame(
                self._env.physics.render(camera_id=0), t
            )
            kwargs["step_callback"] = record_frame
            rollout = self.get_rollout(*args, **kwargs)
            rollout.frames = recorder.frames
            if mp4path is not None:
                recorder.make_movie(mp4path)
        return rollout


class PointMass(EnvWrapper):
    state_dim = 4
    observation_dim = 4

    def get_goal_weights(self) -> torch.Tensor:
        weights = super().get_goal_weights()
        weights[0:2] = 10 * self._state_penalty
        weights[2:] = (
            self._state_penalty / 4.0
        )  # Penalties on the velocities act as dampers
        return weights

    def set_goal(self) -> torch.Tensor:
        target = np.random.uniform(-0.25, 0.25, 3)
        target[-1] = 0.01
        self._env.physics.named.model.geom_pos["target"] = target
        goal_state = torch.zeros(self.state_dim, dtype=torch.float32)
        goal_state[0] = target[0]
        goal_state[1] = target[1]
        return goal_state


class Reacher(EnvWrapper):
    state_dim = 4
    observation_dim = 6
    override_control_timestep = 0.04

    def sample_state(self) -> torch.Tensor:
        state = np.zeros(self.state_dim)

        state[0] = np.random.uniform(low=-np.pi, high=np.pi)
        state[1] = np.random.uniform(low=-2.8, high=2.8)  # Avoid infeasible goals
        state[2] = np.random.uniform(low=-3, high=3)
        state[3] = np.random.uniform(low=-3, high=3)

        return torch.tensor(state, dtype=torch.float32)

    def get_goal_weights(self) -> torch.Tensor:
        weights = torch.zeros(self.observation_dim)
        weights[0:2] = self._state_penalty
        weights[4:] = 0 # no penalty on the target's location

        weights[2:4] = (
            self._state_penalty / 20.0
        )  # Penalties on the velocities act as dampers
        return weights

    def set_goal_state(self):
        goal_state = torch.zeros(self.state_dim, dtype=torch.float32)
        goal_state[0] = np.random.uniform(low=-np.pi, high=np.pi)
        goal_state[1] = np.random.uniform(low=-2.8, high=2.8)  # Avoid infeasible goals
        goal_state[2] = 0
        goal_state[3] = 0
        return goal_state

    def set_goal_observation(self): 
        goal_state = self.set_goal_state()
        target_x, target_y = Reacher.get_xy(goal_state)
        goal_observation = torch.cat([goal_state, torch.tensor([target_x, target_y])])
        return goal_observation

    def set_goal(self) -> torch.Tensor:
        return self.set_goal_observation()

    def set_target(self, state):
        target_x, target_y = self.get_xy(state)
        self._env.physics.named.model.geom_pos["target", "x"] = target_x
        self._env.physics.named.model.geom_pos["target", "y"] = target_y

    @staticmethod
    def get_xy(goal_state):
        a = 0.12 * np.cos(goal_state[1])
        b = 0.12 * np.sin(goal_state[1])
        theta = goal_state[0] + np.arctan(b / (0.12 + a))
        mag = np.sqrt((0.12 + a) ** 2 + b ** 2)
        target_x = mag * np.cos(theta)
        target_y = mag * np.sin(theta)
        return target_x, target_y

    def sample_rollouts_biased_rewards(self, num_rollouts=20, num_steps=100):
        rollouts = []
        for _ in range(num_rollouts):
            initial_state = self.set_goal_state()
            rollout = self.get_rollout(
                num_steps=num_steps,
                set_state=True,
                goal_state=initial_state,
                initial_state=initial_state
            )
            rollouts.append(rollout)
        return rollouts

class Cheetah(EnvWrapper):
    state_dim = 18 - 1 + 2
    observation_dim = 17

    def sample_state(self) -> torch.Tensor:
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

        return torch.tensor(state, dtype=torch.float32)

    def get_state(self) -> torch.Tensor:
        state = super().get_state()[1:]
        state = np.append(state, self._env.physics.speed())  # Add horizontal speed
        state = np.append(
            state, self._env.physics.named.data.subtree_com["torso"][2]
        )  # Add torso height
        return torch.tensor(state, dtype=torch.float32)

    def get_goal_weights(self) -> torch.Tensor:
        weights = super().get_goal_weights()
        weights[17] = self._state_penalty
        weights[18] = self._state_penalty / 2.0
        return weights

    def set_goal(self) -> torch.Tensor:
        goal_state = torch.zeros(self.state_dim, dtype=torch.float32)
        goal_state[-2] = 2  # Target Speed
        goal_state[-1] = 0.4  # Target torso height
        return goal_state


class Manipulator(EnvWrapper):
    state_dim = 22 + 7
    observation_dim = 37

    def get_state(self) -> torch.Tensor:
        state = super().get_state()
        state = np.append(
            state, self._env.physics.named.data.site_xpos["grasp", "x"]
        )  # gripper position
        state = np.append(
            state, self._env.physics.named.data.site_xpos["grasp", "z"]
        )  # gripper position
        state = np.append(state, self._env.physics.touch())  # Sensors. 5 dimensions
        return torch.tensor(state, dtype=torch.float32)

    def get_goal_weights(self) -> torch.Tensor:
        weights = super().get_goal_weights()
        weights[8:10] = 10 * self._state_penalty
        weights[10:21] = self._state_penalty / 4
        weights[-7:-5] = 10 * self._state_penalty
        weights[-5:] = self._state_penalty / 20
        return weights

    def set_goal(self) -> torch.Tensor:
        goal_state = torch.zeros(self.state_dim, dtype=torch.float32)
        target_ball = self._env.physics.body_location("target_ball")[:2]
        # Want the ball to be over the target
        goal_state[8] = target_ball[0]
        goal_state[9] = target_ball[1]
        # Want the gripper location to be at the target
        goal_state[-7] = target_ball[0]
        goal_state[-6] = target_ball[1]
        goal_state[-5:] = 0.5  # Contact sensors
        return goal_state


class Humanoid(EnvWrapper):
    state_dim = 55 + 5
    observation_dim = 67

    def sample_state(self) -> torch.Tensor:
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

        return torch.tensor(state, dtype=torch.float32)

    def sample_action(self, batch_size=None) -> torch.Tensor:
        if batch_size is None:
            action = np.random.normal(0, 0.4, self.action_dim)
            action[3:-6] = 0.0
        else:
            action = np.random.normal(0, 0.4, self.action_dim * batch_size).reshape(
                (batch_size, -1)
            )
            action[:, 3:-6] = 0.0
        return torch.tensor(action, dtype=torch.float32)

    def get_state(self) -> torch.Tensor:
        state = super().get_state()  # 55 (pure state)
        # state = np.append(state, self.env.physics.head_height()) ## +1 head height (target should be 1.6)
        # state = np.append(state, self.env.physics.torso_upright()) ## +1 torso projected height (target should be 1.0)
        com_pos = self.env.physics.center_of_mass_position()
        rfoot = self.env.physics.named.data.xpos["right_foot"]
        lfoot = self.env.physics.named.data.xpos["left_foot"]
        ave_foot = (rfoot + lfoot) / 2.0
        above_feet = ave_foot + np.array([0.0, 0.0, 1.3])
        torso = self.env.physics.named.data.xpos["torso"]

        first_penalty = np.linalg.norm(com_pos[:2] - ave_foot[:2])  # First described by Tassa
        second_penalty = np.linalg.norm(com_pos[:2] - torso[:2])  # Second term described by Tassa
        third_penalty = np.linalg.norm(torso[1:] - above_feet[1:])  # Third term

        state = np.append(state, first_penalty)  # +1
        state = np.append(state, second_penalty)  # +1
        state = np.append(state, third_penalty)  # +1
        state = np.append(
            state, self.env.physics.center_of_mass_velocity()[:2]
        )  # +2 velocity should be 0
        return torch.tensor(state, dtype=torch.float32)

    def get_goal_weights(self) -> torch.Tensor:
        weights = super().get_goal_weights()
        weights[-5:] = 10 * self._state_penalty
        return weights

    def set_goal(self) -> torch.Tensor:
        goal_state = torch.zeros(self.state_dim, dtype=torch.float32)
        return goal_state


class Swimmer(EnvWrapper):
    state_dim = 10 + 2

    def sample_state(self) -> torch.Tensor:
        state_dim = 10
        # Assuming swimmer3
        state = np.zeros(state_dim)

        state[2] = np.random.uniform(low=-3, high=3)

        return torch.tensor(state, dtype=torch.float32)

    def get_state(self) -> torch.Tensor:
        state = super().get_state()  # 16-2 dims
        state = np.append(
            state, self.env.physics.named.data.xmat["head"][:2]
        )  # Head orientation (2)
        return torch.tensor(state, dtype=torch.float32)

    def get_goal_weights(self) -> torch.Tensor:
        weights = super().get_goal_weights()
        weights[0:1] = 10 * self._state_penalty  # Distance to target
        # weights[18] = self._state_penalty/2.
        weights[5:-2] = self._state_penalty  # velocities
        # weights[8:-2] = self._state_penalty/4. ## velocities
        return weights

    def set_goal(self) -> torch.Tensor:
        target = self._env.physics.named.data.geom_xpos["target"][:2]
        goal_state = torch.zeros(self.state_dim, dtype=torch.float32)
        goal_state[0] = target[0]
        goal_state[1] = target[1]
        return goal_state


class Walker(EnvWrapper):
    state_dim = 18 - 1 + 3
    observation_dim = 24

    def sample_state(self) -> torch.Tensor:
        state_dim = 18
        state = np.zeros(state_dim)
        # state[0] = np.random.uniform() ## horizontal position
        # state[1] = 0#np.random.uniform(0, 0.05) ## vertical position
        state[2] = np.random.uniform(-0.1, 0.1)  # clock rotation of main body

        hip_rot = np.random.uniform(-0.15, 0.15)
        state[3] = hip_rot  # right_hip (-20, 100)   = (-0.3491, 1.7452)
        state[4] = np.random.uniform(-0.3, 0)  # right_knee (-150, 0)   = (-2.6178 , 0)
        state[5] = np.random.uniform(-0.1, 0.1)  # right_ankle (-45, 45)  = (-0.7854, 0.7854)

        state[6] = -hip_rot
        state[7] = np.random.uniform(-0.3, 0)  # left_knee (-150, 0)   = (-2.6178 , 0)
        state[8] = np.random.uniform(-0.1, 0.1)  # left_ankle (-45, 45)  = (-0.7854, 0.7854)

        # state[9:] = np.random.uniform(-0.04, 0.04, 9) # Velocities

        return torch.tensor(state, dtype=torch.float32)

    def get_state(self) -> torch.Tensor:
        state = super().get_state()[1:]
        state = np.append(state, self.env.physics.torso_upright())  # Add torso upright
        state = np.append(state, self.env.physics.torso_height())  # Add torso height
        state = np.append(
            state, self.env.physics.horizontal_velocity()
        )  # Add horizontal speed
        return torch.tensor(state, dtype=torch.float32)

    def get_goal_weights(self) -> torch.Tensor:
        weights = super().get_goal_weights()
        weights[-3:] = self._state_penalty
        return weights

    def set_goal(self) -> torch.Tensor:
        goal_state = torch.zeros(self.state_dim, dtype=torch.float)
        goal_state[-3] = 1.0  # target torso_upright
        goal_state[-2] = 1.3  # target torso_height
        goal_state[-1] = 3.0  # target speed
        return goal_state


class Hopper(EnvWrapper):
    state_dim = 14 - 1 + 4
    observation_dim = 15

    def sample_state(self) -> torch.Tensor:
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

        return torch.tensor(state, dtype=torch.float32)

    def get_state(self) -> torch.Tensor:
        state = super().get_state()[1:]  # Removed horizontal position
        state = np.append(state, self.env.physics.touch())  # Add touch sensors in feet
        state = np.append(state, self.env.physics.height())  # Add torso height
        state = np.append(state, self.env.physics.speed())  # Add horizontal speed
        return torch.tensor(state, dtype=torch.float32)

    def get_goal_weights(self) -> torch.Tensor:
        weights = super().get_goal_weights()
        weights[-2] = self._state_penalty / 2.0
        weights[-1] = self._state_penalty
        return weights

    def set_goal(self) -> torch.Tensor:
        goal_state = torch.zeros(self.state_dim, dtype=torch.float)
        goal_state[-2] = 0.9  # target torso_height
        goal_state[-1] = 1.0  # target speed
        return goal_state

