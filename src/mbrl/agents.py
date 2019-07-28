import torch
from torch.nn import Module as TorchNNModule
from src.mbrl.data import Rollout, TransitionsDataset
from src.mbrl.env_wrappers import EnvWrapper
from src.mbrl.planners import ModelPlanner, ScalarTorchFunc
from src.mbrl.models import DynamicsModel

from typing import List


class Agent:
    def __init__(
        self,
        environment: EnvWrapper,
        planner: ModelPlanner,
        model: DynamicsModel,
        horizon: int,
        action_cost: ScalarTorchFunc,
        state_cost: ScalarTorchFunc,
        rollout_length: int,
        num_rollouts_per_iteration: int,
        num_train_iterations: int,
        num_epochs_per_iteration: int,
        batch_size: int,
    ):
        self.environment = environment
        self.planner = planner
        self.model = model
        self.horizon = horizon
        self.action_cost = action_cost
        self.state_cost = state_cost
        self.rollout_length = rollout_length
        self.num_rollouts_per_iteration = num_rollouts_per_iteration
        self.num_train_iterations = num_train_iterations
        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.batch_size = batch_size
        self.dataset = TransitionsDataset(rollouts=[], transitions_capacity=100000)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def add_rollouts(self, get_action=None):
        rollouts = [
            self.environment.get_rollout(self.rollout_length, get_action)
            for i in range(self.num_rollouts_per_iteration)
        ]
        self.dataset.add_rollouts(rollouts)

    def train(self):
        self.add_rollouts()
        for iteration in range(self.num_train_iterations):
            self.model.train_model(dataset=self.dataset, batch_size=self.batch_size, num_epochs=self.num_epochs_per_iteration)
            self.add_rollouts(get_action=self.get_action)


class MPCAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_trajectory = None

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        if self.last_trajectory is not None:
            initial_trajectory = (
                self.last_trajectory[0][1:],
                self.last_trajectory[1][1:],
            )
        else:
            initial_trajectory = None
        self.last_trajectory = self.planner.plan(
            initial_state=state,
            model=self.model,
            state_cost=self.state_cost,
            action_cost=self.action_cost,
            sample_action=self.environment.sample_action,
            horizon=self.horizon,
            initial_trajectory=initial_trajectory,
        )
        return self.last_trajectory[1][0]
