from typing import List, Optional
from collections import defaultdict
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from enum import Enum

class Rollout:
    def __init__(self, states, observations, actions, rewards):
        assert len(states) > 0
        assert len(states) == len(observations)
        assert len(actions) == len(rewards)
        assert len(states) == (len(actions) + 1)

        self._length = len(rewards)
        self._states = states

        self._flat_obs = isinstance(observations[0], torch.Tensor)
        self._observations = observations
        self._actions = actions + [None]
        self._rewards = [None] + rewards
        self._rollout = list(
            zip(self.rewards, self.states, self.observations, self.actions)
        )
        self._flat_roll = self._to_array()

    @property
    def states(self):
        return self._states

    @property
    def observations(self):
        return self._observations

    @property
    def actions(self):
        return self._actions

    @property
    def rewards(self):
        return self._rewards

    @property
    def sum_of_rewards(self):
        return sum(self.rewards[1:])

    @property
    def flat_observations(self):
        return self._flat_obs

    def __len__(self):
        return self._length

    def __getitem__(self, key):
        return Rollout(
            states=self.states[key],
            observations=self.observations[key],
            actions=self.actions[key][:-1],
            rewards=self.rewards[key][1:],
        )

    def __repr__(self):
        return str(self._rollout)

    def get_transition(self, idx, stats=None):
        """
        Returns a tuple of tuples of the form (s_i, o_i, a_i), (r_{i+1}, s_{i+1}, o_{i+1})
        """
        assert idx < len(self)

        inputs = self._flat_roll[4 * idx : 4 * idx + 3]
        outputs = self._flat_roll[4 * idx + 3 : 4 * idx + 6]

        if stats is not None:
            inputs[0] = (inputs[0] - stats["states"]["mean"]) / stats["states"]["std"]

            if self.flat_observations:
                inputs[1] = (inputs[1] - stats["observations"]["mean"]) / stats["observations"]["std"]
            else:
                normalised_obs = {}
                for k, v in inputs[1].items():
                    normalised_obs[k] = (v - stats["observations"][k]["mean"]) / stats[
                        "observations"
                    ][k]["std"]
                inputs[1] = normalised_obs

            inputs[2] = (inputs[2] - stats["actions"]["mean"]) / stats["actions"]["std"]

            outputs[0] = (outputs[0] - stats["rewards"]["mean"]) / stats["rewards"][
                "std"
            ]
            outputs[1] = (outputs[1] - stats["states"]["mean"]) / stats["states"]["std"]

            if self.flat_observations:
                outputs[2] = (outputs[2] - stats["observations"]["mean"]) / stats["observations"]["std"]
            else:
                normalised_obs = {}
                for k, v in outputs[2].items():
                    normalised_obs[k] = (v - stats["observations"][k]["mean"]) / stats[
                        "observations"
                    ][k]["std"]
                outputs[2] = normalised_obs

        return tuple(inputs), tuple(outputs)

    def get_multistep_transitions(self, start_idx, horizon, stats=None):
        assert start_idx + horizon < len(self) + 1
        inputs, outputs = [], []
        for i in range(start_idx, start_idx + horizon):
            inp, out = self.get_transition(i, stats=stats)
            inputs.append(inp)
            outputs.append(out)

        return tuple(inputs), tuple(outputs)

    def _to_array(self):
        # [s_0, o_0, a_0, r_1, s_1, o_1, a_1, ... , r_{K-1}, s_{K-1}, o_{K-1}, a_{K-1}, r_K, s_K, o_K]
        rollout = []
        for r in self._rollout:
            rollout += list(r)

        return rollout[1:-1]

class TransitionsDatasetDataMode(Enum):
    state_only = 'state_only'
    obs_only = 'obs_only'
    both = 'both'
class TransitionsDataset(Dataset):
    def __init__(
        self,
        rollouts: Optional[List[Rollout]] = None,
        transitions_capacity: int = int(1e6),
        horizon: int = 1,
        normalise=True,
        data_mode=TransitionsDatasetDataMode.both
    ):
        super().__init__()
        self.capacity = transitions_capacity
        self.horizon = horizon
        self._rollouts = []
        self._occupied_capacity = 0
        self._flat_obs = None
        self._stats = {
            "state": None,
            "observation": None,
            "action": None,
            "reward": None,
        }
        self._normalise = normalise
        self._data_mode = data_mode

        # This check exists so that an empty dataset can be instantiated
        if rollouts is not None:
            self.add_rollouts(rollouts)
    
    def set_data_mode(self, data_mode):
        self._data_mode = TransitionsDatasetDataMode(data_mode)

    def add_rollouts(self, rollouts):
        if self._flat_obs is None:
            # The transitions dataset is gonna assume all rollouts have observations like the first one
            self._flat_obs = rollouts[0].flat_observations

        for roll in rollouts:
            assert roll.flat_observations == self._flat_obs
            self._occupied_capacity += max(len(roll) - self.horizon + 1, 0)
            self._rollouts.append(roll)

        over_capacity = self._occupied_capacity - self.capacity
        if over_capacity > 0:
            print("Exceeded max_capacity of {} transitions".format(self.capacity))
            print("Removing oldest {} transitions from dataset".format(over_capacity))
            while over_capacity > 0:
                oldest_roll_len = len(self._rollouts[0])
                if oldest_roll_len > over_capacity:
                    self._rollouts[0] = self._rollouts[0][over_capacity:]
                    self._occupied_capacity = self.capacity
                    break
                else:
                    _ = self._rollouts.pop(0)
                    self._occupied_capacity -= oldest_roll_len
                    over_capacity = self._occupied_capacity - self.capacity

        self._update_stats()

    @property
    def occupied_capacity(self):
        return sum([len(r) - self.horizon + 1 for r in self._rollouts])

    @property
    def num_rollouts(self):
        return len(self._rollouts)

    @property
    def statistics(self):
        return self._stats

    @property
    def rollouts(self):
        return self._rollouts

    def __len__(self):
        return self._occupied_capacity

    def __getitem__(self, trans):
        return self.get_transition(roll_idx=trans[0], start_idx=trans[1])

    def get_transition(self, roll_idx=None, start_idx=None):
        if roll_idx is None:
            roll_idx = np.random.randint(0, self.num_rollouts)
        rollout = self._rollouts[roll_idx]
        if start_idx is None:
            start_idx = np.random.randint(0, len(rollout) - self.horizon)

        stats = self._stats if self._normalise else None
        (inputs, outputs) = rollout.get_multistep_transitions(
            start_idx, self.horizon, stats=stats
        )
        if self._data_mode == TransitionsDatasetDataMode.state_only:
            inputs = [ inp[::2] for inp in inputs]
            outputs =  [outp[:-1] for outp in outputs]
        elif self._data_mode == TransitionsDatasetDataMode.obs_only:
            inputs = [ inp[1::] for inp in inputs]
            outputs =  [outp[::2] for outp in outputs]
        elif self._data_mode is not TransitionsDatasetDataMode.both:
            raise ValueError('Unknown mode.')
        return inputs, outputs
        
    def _update_stats(self):
        stats = {}
        all_states, all_actions, all_rewards = [], [], []
        all_obs = [] if self._flat_obs else defaultdict(lambda: [])
        for r in self._rollouts:
            all_states.append(torch.stack(r.states))
            all_actions.append(torch.stack(r.actions[:-1]))  # Ignore last action
            all_rewards.append(torch.stack(r.rewards[1:]))  # Ignore first reward
            if self._flat_obs:
                all_obs.append(torch.stack(r.observations))
            else:
                for obs in r.observations:
                    for k, v in obs.items():
                        all_obs[k].append(v)

        stats["states"] = self._get_stats(torch.cat(all_states))
        stats["actions"] = self._get_stats(torch.cat(all_actions))
        stats["rewards"] = self._get_stats(torch.cat(all_rewards))

        if self._flat_obs:
            stats["observations"] = self._get_stats(torch.cat(all_obs))
        else:
            stats["observations"] = {
                k: self._get_stats(torch.stack(v)) for k, v in all_obs.items()
            }

        self._stats = stats

    def unnormalize_state(self, state):
        if not self._normalise:
            return state
        # (outputs[1] - stats["states"]["mean"]) / stats["states"]["std"]
        return (state * self._stats["states"]["std"]) + self._stats["states"]["mean"]

    def normalize_state(self, state):
        if not self._normalise:
            return state
        return (state - self._stats["states"]["mean"]) / self._stats["states"]["std"]

    def unnormalize_obs(self, obs):
        if not self._normalise:
            return obs
        return (obs * self._stats["observations"]["std"]) + self._stats["observations"]["mean"]

    def normalize_obs(self, obs):
        if not self._normalise:
            return obs
        return (obs - self._stats["observations"]["mean"]) / self._stats["observations"]["std"]

    def unnormalize_action(self, action):
        if not self._normalise:
            return action
        return (action * self._stats["actions"]["std"]) + self._stats["actions"]["mean"]

    def normalize_action(self, action):
        if not self._normalise:
            return action
        return (action - self._stats["actions"]["mean"]) / self._stats["actions"]["std"]

    def unnormalize_reward(self, reward):
        if not self._normalise:
            return reward
        return (reward * self._stats["rewards"]["std"]) + self._stats["rewards"]["mean"]

    def normalize_reward(self, reward):
        if not self._normalise:
            return reward
        return (reward - self._stats["rewards"]["mean"]) / self._stats["rewards"]["std"]

    @staticmethod
    def _get_stats(array):
        return {
            "mean": torch.mean(array, dim=0),
            "std": torch.std(array, dim=0),
            "min": torch.min(array, dim=0).values,
            "max": torch.max(array, dim=0).values,
        }


class TransitionsSampler(Sampler):
    def __init__(self, data_source: TransitionsDataset):
        self.data_source = data_source

    def __iter__(self):
        possible_transitions = []
        for roll_idx, roll in enumerate(self.data_source.rollouts):
            for start_idx in range(len(roll) - self.data_source.horizon):
                possible_transitions.append((roll_idx, start_idx))
        np.random.shuffle(possible_transitions)

        for trans in possible_transitions:
            yield trans


# -------------------------------------------------------------------------------------------
# Some things for testing
# state_dim = 2
# obs_dim = 2
# act_dim = 1
#
# h_1 = 50000
# state_dist = torch.distributions.Normal(loc=150, scale=50)
# first_dist = torch.distributions.Normal(loc=0, scale=15)
# second_dist = torch.distributions.Normal(loc=-12, scale=0.3)
# act_dist = torch.distributions.Normal(loc=1, scale=2)
# reward_dist = torch.distributions.Normal(loc=0, scale=10)
#
# s = [state_dist.sample((state_dim,)) for _ in range(h_1)]
# o = [{'first': first_dist.sample(), 'second': second_dist.sample()} for _ in range(h_1)]
# a = [act_dist.sample((act_dim,)) for _ in range(h_1 - 1)]
# r = [reward_dist.sample() for _ in range(h_1 - 1)]
# r1 = Rollout(states=s, observations=o, actions=a, rewards=r)
#
# h_2 = 12000
# s = [state_dist.sample((state_dim,)) for _ in range(h_2)]
# o = [{'first': first_dist.sample(), 'second': second_dist.sample()} for _ in range(h_2)]
# a = [act_dist.sample((act_dim,)) for _ in range(h_2 - 1)]
# r = [reward_dist.sample() for _ in range(h_2 - 1)]
# r2 = Rollout(states=s, observations=o, actions=a, rewards=r)
#
# h_3 = 40000
# s = [state_dist.sample((state_dim,)) for _ in range(h_3)]
# o = [{'first': first_dist.sample(), 'second': second_dist.sample()} for _ in range(h_3)]
# a = [act_dist.sample((act_dim,)) for _ in range(h_3 - 1)]
# r = [reward_dist.sample() for _ in range(h_3 - 1)]
# r3 = Rollout(states=s, observations=o, actions=a, rewards=r)
#
# dataset = TransitionsDataset(rollouts=[r1, r2], transitions_capacity=100000, horizon=2)
#
# transitions = [t for t in dataset]
# inputs, targets = dataset.get_transition()
#
# dataset.add_rollouts([r3])
#
#
# print("-----")
