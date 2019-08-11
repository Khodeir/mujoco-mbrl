import os
import torch
import pickle

from src.mbrl.data import TransitionsDataset, TransitionsDatasetDataMode
from src.mbrl.env_wrappers import EnvWrapper
from src.mbrl.planners import ModelPlanner
from src.mbrl.models import DynamicsModel, ModelWithReward
from functools import partial
from operator import itemgetter

import numpy as np
from src.mbrl.logger import logger
from tensorboardX import SummaryWriter
from typing import Callable, Dict, Optional

ScalarTorchFunc = Callable[[torch.Tensor], float]


def save(agent, path):
    writer = agent.writer
    agent.writer = None
    with open(path, "wb") as f:
        pickle.dump(agent, f)
    agent.writer = writer


class MPCAgent:
    def __init__(
        self,
        environment: EnvWrapper,
        planner: ModelPlanner,
        model: DynamicsModel,
        horizon: int,
        optimizer: torch.optim.Optimizer,
        rollout_length: int,
        num_rollouts_per_iteration: int,
        num_train_iterations: int,
        writer: SummaryWriter,
        base_path: str,
        dataset: Optional[TransitionsDataset] = None
    ):
        self.environment = environment
        self.planner = planner
        self.model = model
        self.horizon = horizon
        self.rollout_length = rollout_length
        self.num_rollouts_per_iteration = num_rollouts_per_iteration
        self.num_train_iterations = num_train_iterations
        self.optimizer = optimizer
        self.dataset = TransitionsDataset(transitions_capacity=10000) if dataset is None else dataset
        self.writer = writer or SummaryWriter()
        self.train_iterations = 0
        self.last_trajectory = None
        self.base_path = base_path
        self.training_goal_state = None
        self.num_initial_rollouts = 20

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def _add_rollouts(
        self, get_action=None, num_rollouts=None, set_state=False, record_last=True, override_goal_state=None, override_initial_state=None
    ):
        rollout_type = "policy" if get_action else "random"
        logger.info(
            "Generating {} {} rollouts of {} length.".format(
                self.num_rollouts_per_iteration, rollout_type, self.rollout_length
            )
        )
        rollouts = []
        num_rollouts = num_rollouts or self.num_rollouts_per_iteration
        for i in range(num_rollouts):
            self.last_trajectory = None
            if record_last and i == num_rollouts - 1:
                rollouts.append(
                    self.environment.record_rollout(
                        num_steps=self.rollout_length,
                        get_action=get_action,
                        mp4path=os.path.join(self.base_path, "last_rollout_{}".format(self.train_iterations)),
                        set_state=set_state,
                        goal_state=self.training_goal_state if override_goal_state is None else override_goal_state,

                    )
                )
                continue
            rollouts.append(
                self.environment.get_rollout(
                    self.rollout_length, get_action, set_state=set_state, goal_state=self.training_goal_state, initial_state=override_initial_state
                )
            )

        self.dataset.add_rollouts(rollouts)
        self._record_metrics(rollouts, rollout_type)

    def _record_metrics(self, rollouts, rollout_type):
        sum_rewards = [rollout.sum_of_rewards for rollout in rollouts]
        self.writer.add_scalar(
            "AvgRolloutRewards/{}".format(rollout_type),
            np.mean(sum_rewards),
            self.train_iterations,
        )
        self.writer.add_histogram(
            "RolloutRewards/{}".format(rollout_type), sum_rewards, self.train_iterations
        )
        # I think this gif generator is broken
        # for rollout in rollouts:
        #     if hasattr(rollout, "frames"):
        #         self.writer.add_video(
        #             "LastRollout/{}".format(rollout_type),
        #             torch.stack(
        #                 [
        #                     torch.from_numpy(np.array(f).transpose(2, 0, 1))
        #                     for f in rollout.frames
        #                 ],
        #                 dim=0,
        #             ).unsqueeze(0),
        #             self.train_iterations,
        #         )


class GoalStateAgent(MPCAgent):
    def __init__(
        self,
        environment: EnvWrapper,
        planner: ModelPlanner,
        model: DynamicsModel,
        horizon: int,
        optimizer: torch.optim.Optimizer,
        rollout_length: int,
        num_rollouts_per_iteration: int,
        num_train_iterations: int,
        writer: SummaryWriter,
        action_cost: ScalarTorchFunc,
        state_cost: ScalarTorchFunc,
        base_path: str,
        dataset: Optional[TransitionsDataset] = None
    ):
        super().__init__(
            environment=environment,
            planner=planner,
            model=model,
            horizon=horizon,
            optimizer=optimizer,
            rollout_length=rollout_length,
            num_rollouts_per_iteration=num_rollouts_per_iteration,
            num_train_iterations=num_train_iterations,
            writer=writer,
            base_path=base_path,
            dataset=dataset,
        )

        self.action_cost = action_cost
        self.state_cost = state_cost

        self.dataset.set_data_mode(TransitionsDatasetDataMode.obs_only)
        self.normalize_state = self.dataset.normalize_obs
        self.unnormalize_state = self.dataset.unnormalize_obs
        self.normalize_action = self.dataset.normalize_action

    def _reset_goal(self):
        self.training_goal_state = self.environment.set_goal()
        self.state_cost.set_goal_state(self.training_goal_state)

    def _record_metrics(self, rollouts, rollout_type):
        super()._record_metrics(rollouts, rollout_type)
        state_costs = [
            sum(map(self.state_cost, rollout.observations)) for rollout in rollouts
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
            sum(map(self.action_cost, rollout.actions[:-1])) for rollout in rollouts
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
        self._reset_goal()
        self._add_rollouts(num_rollouts=self.num_initial_rollouts)
        for iteration in range(1, self.num_train_iterations + 1):
            logger.debug("Iteration {}".format(iteration))
            # TODO: Check if we need to alter the frequency of reset goal

            self._reset_goal()
            self.train_iterations = iteration
            self.model.train_model(
                dataset=self.dataset, optimizer=self.optimizer, writer=self.writer
            )
            self._add_rollouts(get_action=self.get_action)

    def get_action(self, state_and_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # logger.debug('Planning a step. Horizon:{}'.format(self.horizon))
        if self.last_trajectory is not None:
            initial_trajectory = (
                self.last_trajectory[0][1:],
                self.last_trajectory[1][0:],
            )
        else:
            initial_trajectory = None
        self.last_trajectory = self.planner.plan(
            initial_state=state_and_obs["observation"],
            model=partial(
                self.model,
                normalize_state=self.normalize_state,
                normalize_action=self.normalize_action,
                unnormalize_state=self.unnormalize_state,
            ),
            cost=lambda state, action: self.state_cost(state)
            + self.action_cost(action),
            sample_action=self.environment.sample_action,
            horizon=self.horizon,
            initial_trajectory=initial_trajectory,
        )
        return self.last_trajectory[1][0].flatten()


class RewardAgent(MPCAgent):
    def __init__(
        self,
        environment: EnvWrapper,
        planner: ModelPlanner,
        model: ModelWithReward,
        horizon: int,
        optimizer: torch.optim.Optimizer,
        rollout_length: int,
        num_rollouts_per_iteration: int,
        num_train_iterations: int,
        writer: SummaryWriter,
        base_path: str,
        dataset: Optional[TransitionsDataset] = None
    ):
        super().__init__(
            environment=environment,
            planner=planner,
            model=model,
            horizon=horizon,
            optimizer=optimizer,
            rollout_length=rollout_length,
            num_rollouts_per_iteration=num_rollouts_per_iteration,
            num_train_iterations=num_train_iterations,
            writer=writer,
            base_path=base_path,
            dataset=dataset
        )
        self.dataset.set_data_mode(TransitionsDatasetDataMode.obs_only)
        self.normalize_state = self.dataset.normalize_obs
        self.unnormalize_state = self.dataset.unnormalize_obs
        self.normalize_action = self.dataset.normalize_action
        self.normalize_reward = self.dataset.normalize_reward
        self.unnormalize_reward = self.dataset.unnormalize_reward

    def train(self):
        logger.info("Starting outer training loop.")
        self._add_rollouts(num_rollouts=self.num_initial_rollouts)
        for iteration in range(1, self.num_train_iterations + 1):
            logger.debug("Iteration {}".format(iteration))
            self.train_iterations = iteration
            self.model.train_model(
                dataset=self.dataset, optimizer=self.optimizer, writer=self.writer
            )
            self._add_rollouts(get_action=self.get_action)

    def get_action(self, state_and_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        def compose(a, b):
            def ab(*args, **kwargs):
                return b(a(*args, **kwargs))

            return ab

        # logger.debug('Planning a step. Horizon:{}'.format(self.horizon))

        if self.last_trajectory is not None:
            initial_trajectory = (
                self.last_trajectory[0][1:],
                self.last_trajectory[1][0:],
            )
        else:
            initial_trajectory = None
        self.last_trajectory = self.planner.plan(
            initial_state=state_and_obs["observation"],
            model=compose(
                partial(
                    self.model,
                    normalize_state=self.normalize_state,
                    normalize_action=self.normalize_action,
                    unnormalize_state=self.unnormalize_state,
                    unnormalize_reward=self.unnormalize_reward,
                ),
                itemgetter(0),
            ),
            cost=compose(
                partial(
                    self.model,
                    normalize_state=self.normalize_state,
                    normalize_action=self.normalize_action,
                    unnormalize_state=self.unnormalize_state,
                    unnormalize_reward=self.unnormalize_reward,
                ),
                itemgetter(1),
            ),
            sample_action=self.environment.sample_action,
            horizon=self.horizon,
            initial_trajectory=initial_trajectory,
        )
        return self.last_trajectory[1][0].flatten()

