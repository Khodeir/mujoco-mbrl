import torch
import pickle

from src.mbrl.data import TransitionsDataset
from src.mbrl.env_wrappers import EnvWrapper
from src.mbrl.planners import ModelPlanner, ScalarTorchFunc
from src.mbrl.models import DynamicsModel
from functools import partial

import numpy as np
from src.mbrl.logger import logger
from tensorboardX import SummaryWriter


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
        writer: SummaryWriter,
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
        self.optimizer = optimizer
        self.dataset = TransitionsDataset(rollouts=[], transitions_capacity=100000)
        self.writer = writer or SummaryWriter()
        self.train_iterations = 0

    def reset_goal(self):
        goal_state = self.environment.set_goal()
        self.state_cost.set_goal_state(goal_state)

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def add_rollouts(self, get_action=None):
        rollout_type = "policy" if get_action else "random"
        logger.info(
            "Generating {} {} rollouts of {} length.".format(
                self.num_rollouts_per_iteration, rollout_type, self.rollout_length
            )
        )
        rollouts = [
            self.environment.get_rollout(self.rollout_length, get_action)
            for i in range(self.num_rollouts_per_iteration)
        ]
        self.dataset.add_rollouts(rollouts)

        sum_rewards = [rollout.sum_of_rewards for rollout in rollouts]
        self.writer.add_scalar(
            "AvgRolloutRewards/{}".format(rollout_type),
            np.mean(sum_rewards),
            self.train_iterations,
        )
        self.writer.add_histogram(
            "RolloutRewards/{}".format(rollout_type), sum_rewards, self.train_iterations
        )

        state_costs = [
            rollout.get_sum_of_state_costs(self.state_cost) for rollout in rollouts
        ]
        self.writer.add_scalar(
            "AvgRolloutStateCosts/{}".format(rollout_type),
            np.mean(state_costs),
            self.train_iterations,
        )
        self.writer.add_histogram(
            "RolloutStateCosts/{}".format(rollout_type),
            state_costs,
            self.train_iterations,
        )

        action_costs = [
            rollout.get_sum_of_action_costs(self.action_cost) for rollout in rollouts
        ]
        self.writer.add_scalar(
            "AvgRolloutActionCosts/{}".format(rollout_type),
            np.mean(action_costs),
            self.train_iterations,
        )
        self.writer.add_histogram(
            "RolloutActionCosts/{}".format(rollout_type),
            action_costs,
            self.train_iterations,
        )
        total_costs = np.add(action_costs, state_costs)
        self.writer.add_scalar(
            "AvgRolloutTotalCosts/{}".format(rollout_type),
            np.mean(total_costs),
            self.train_iterations,
        )
        self.writer.add_histogram(
            "RolloutTotalCosts/{}".format(rollout_type),
            total_costs,
            self.train_iterations,
        )

    def train(self):
        logger.info("Starting outer training loop.")
        self.reset_goal()
        self.add_rollouts()
        for iteration in range(1, self.num_train_iterations + 1):
            logger.debug("Iteration {}".format(iteration))
            # TODO: Check if we need to alter the frequency of reset goal
            self.reset_goal()
            self.train_iterations = iteration
            self.model.train_model(
                dataset=self.dataset, optimizer=self.optimizer, writer=self.writer
            )
            self.add_rollouts(get_action=self.get_action)

    @staticmethod
    def save(agent, path):
        writer = agent.writer
        agent.writer = None
        with open(path, "wb") as f:
            pickle.dump(agent, f)
        agent.writer = writer


class MPCAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_trajectory = None

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        # logger.debug('Planning a step. Horizon:{}'.format(self.horizon))
        if self.last_trajectory is not None:
            initial_trajectory = (
                self.last_trajectory[0][1:],
                self.last_trajectory[1][0:],
            )
        else:
            initial_trajectory = None
        self.last_trajectory = self.planner.plan(
            initial_state=state,
            model=partial(
                self.model,
                normalize_state=self.dataset.normalize_state,
                normalize_action=self.dataset.normalize_action,
                unnormalize=self.dataset.unnormalize_state,
            ),
            state_cost=self.state_cost,
            action_cost=self.action_cost,
            sample_action=self.environment.sample_action,
            horizon=self.horizon,
            initial_trajectory=initial_trajectory,
        )
        return self.last_trajectory[1][0].flatten()
