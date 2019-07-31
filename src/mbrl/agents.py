import torch
from torch.nn import Module as TorchNNModule
from src.mbrl.data import Rollout, TransitionsDataset
from src.mbrl.env_wrappers import EnvWrapper
from src.mbrl.planners import ModelPlanner, ScalarTorchFunc
from src.mbrl.models import DynamicsModel
from functools import partial
from typing import List
import numpy as np
from src.mbrl.logger import logger
class Agent:
    def __init__(
        self,
        environment: EnvWrapper,
        planner: ModelPlanner,
        model: DynamicsModel,
        horizon: int,
        action_cost: ScalarTorchFunc,
        state_cost: ScalarTorchFunc,
        optimizer: torch.optim.Optimizer,
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
        self.optimizer = optimizer
        self.dataset = TransitionsDataset(rollouts=[], transitions_capacity=100000)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def add_rollouts(self, get_action=None):
        logger.info('Generating {} {} rollouts of {} length.'.format(self.num_rollouts_per_iteration,'policy' if get_action else 'random', self.rollout_length))
        rollouts = [
            self.environment.get_rollout(self.rollout_length, get_action)
            for i in range(self.num_rollouts_per_iteration)
        ]
        self.dataset.add_rollouts(rollouts)
        logger.record_tabular('AvgSumRewardPerRollout', np.mean([sum(filter(bool, rollout.rewards)) for rollout in rollouts]))
        logger.record_tabular('AvgSumCostPerRollout', np.mean([sum(map(self.state_cost, rollout.states)) for rollout in rollouts]))

    def train(self):
        logger.info('Starting outer training loop.')
        logger.record_tabular('Itr', 0)
        self.add_rollouts()
        logger.dump_tabular()
        for iteration in range(1, self.num_train_iterations + 1):
            logger.record_tabular('Itr', iteration)
            logger.debug('Iteration {}'.format(iteration))
            logger.debug('Training model with batch size {} for {} epochs'.format(self.batch_size, self.num_epochs_per_iteration))
            self.model.train_model(dataset=self.dataset, optimizer=self.optimizer, batch_size=self.batch_size, num_epochs=self.num_epochs_per_iteration)
            self.add_rollouts(get_action=self.get_action)
            logger.dump_tabular()


class MPCAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_trajectory = None

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        logger.debug('Planning a step. Horizon:{}'.format(self.horizon))
        if self.last_trajectory is not None:
            initial_trajectory = (
                self.last_trajectory[0][1:],
                self.last_trajectory[1][1:],
            )
        else:
            initial_trajectory = None
        self.last_trajectory = self.planner.plan(
            initial_state=state,
            model=partial(self.model, unnormalize=self.dataset.unnormalize_state),
            state_cost=self.state_cost,
            action_cost=self.action_cost,
            sample_action=self.environment.sample_action,
            horizon=self.horizon,
            initial_trajectory=initial_trajectory,
        )
        return self.last_trajectory[1][0]
