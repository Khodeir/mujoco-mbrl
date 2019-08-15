import os
import argparse
from enum import Enum
import torch
import numpy as np
from src.mbrl import agents
from src.mbrl import models
from src.mbrl import planners
from src.mbrl import env_wrappers
from src.mbrl import data
from src.mbrl.logger import logger
from tensorboardX import SummaryWriter


class Planner(Enum):
    RandomShooting = "rs"
    GradientDescent = "grad"

    def __str__(self):
        return self.value

    def construct(self):
        if self is Planner.RandomShooting:
            return planners.RandomShootingPlanner
        if self is Planner.GradientDescent:
            return planners.GradientDescentPlanner


class Model(Enum):
    NeuralNet = "nn"
    Linear = "lin"
    ModelWithReward = "rw"

    def __str__(self):
        return self.value

    def construct(self, environment):
        if self is Model.NeuralNet:
            return models.Model(environment.observation_dim, environment.action_dim)
        if self is Model.Linear:
            return models.LinearModel(environment.observation_dim, environment.action_dim)
        if self is Model.ModelWithReward:
            return models.ModelWithReward(environment.observation_dim, environment.action_dim, hidden_units=50)


class Optimizer(Enum):
    Adam = "adam"
    SGD = "sgd"

    def __str__(self):
        return self.value

    def construct(self, model):
        if self is Optimizer.Adam:
            learn_rate = 0.01
            l2_penalty = 0
            return torch.optim.Adam(
                model.parameters(), lr=learn_rate, weight_decay=l2_penalty
            )
        if self is Optimizer.SGD:
            learn_rate = 0.1
            return torch.optim.SGD(model.parameters(), lr=learn_rate)


def Environment(v):
    try:
        env_name, task_name = v.split("_")
    except:
        raise argparse.ArgumentTypeError("<env_name>_<task_name>")
    return env_wrappers.EnvWrapper.load(env_name, task_name)


class Agent(Enum):
    GoalStateAgent = "gs"
    RewardPredictingAgent = "rw"

    def __str__(self):
        return self.value

    def construct(
        self,
        environment,
        planner,
        model,
        horizon,
        optimizer,
        rollout_length,
        num_rollouts_per_iteration,
        num_train_iterations,
        writer,
        base_path,
    ):
        if self is Agent.GoalStateAgent:
            action_cost = models.CoshLoss()
            state_cost = models.SmoothAbsLoss(
                weights=environment.get_goal_weights(),
                goal_state=None
            )
            agent = agents.GoalStateAgent(
                environment=environment,
                planner=planner,
                model=model,
                horizon=horizon,
                action_cost=action_cost,
                state_cost=state_cost,
                optimizer=optimizer,
                rollout_length=rollout_length,
                num_rollouts_per_iteration=num_rollouts_per_iteration,
                num_train_iterations=num_train_iterations,
                writer=writer,
                base_path=base_path,
                dataset=None
            )
            return agent
        if self is Agent.RewardPredictingAgent:
            if isinstance(environment, env_wrappers.Reacher):
                rollouts = environment.sample_rollouts_biased_rewards(num_rollouts=20, num_steps=config["rollout_length"])
                rollout_rewards = np.array([r.rewards[1:] for r in rollouts])
                logger.debug(
                    'reacher initial rollouts stats:\n\tmin: {}\n\tmean: {}\n\tmax: {}\n\tnonzerocount: {}'.format(
                        (rollout_rewards).min(),
                        (rollout_rewards).mean(),
                        (rollout_rewards).max(),
                        (rollout_rewards > 0).sum()
                    )
                )
                dataset = data.TransitionsDataset(rollouts=rollouts, transitions_capacity=10000)
            else:
                dataset = None
            agent = agents.RewardAgent(
                environment=environment,
                planner=planner,
                model=model,
                horizon=config["horizon"],
                optimizer=optimizer,
                rollout_length=config["rollout_length"],
                num_rollouts_per_iteration=config["num_rollouts_per_iteration"],
                num_train_iterations=config["num_train_iterations"],
                writer=writer,
                base_path=base_path,
                dataset=dataset
            )
            if isinstance(environment, env_wrappers.Reacher):
                agent.num_initial_rollouts = 0
            return agent


CONFIG_DEF = (
    {"name": "exp_dir", "type": str},
    {"name": "agent", "type": Agent, "choices": list(Agent)},
    {"name": "environment", "type": Environment},
    {"name": "planner", "type": Planner, "choices": list(Planner)},
    {"name": "model", "type": Model, "choices": list(Model)},
    {"name": "optimizer", "type": Optimizer, "choices": list(Optimizer), "default": "adam"},
    {"name": "horizon", "type": int, "default": 20},
    {"name": "rollout_length", "type": int, "default": 200},
    {"name": "num_rollouts_per_iteration", "type": int, "default": 5},
    {"name": "num_train_iterations", "type": int, "default": 10},
)


def main(config):
    writer = SummaryWriter(log_dir=config["exp_dir"])
    # TODO: check for existence rather than assume its okay to overwrite
    with open(os.path.join(config["exp_dir"], "config.txt"), "w") as f:
        f.write(str(config))
    logger.setup(config["exp_dir"], os.path.join(config["exp_dir"], "log.txt"), "debug")
    environment = config["environment"]
    planner = config["planner"].construct()
    model = config["model"].construct(environment)
    optimizer = config["optimizer"].construct(model)
    agent = config['agent'].construct(
        environment=environment,
        planner=planner,
        model=model,
        horizon=config["horizon"],
        optimizer=optimizer,
        rollout_length=config["rollout_length"],
        num_rollouts_per_iteration=config["num_rollouts_per_iteration"],
        num_train_iterations=config["num_train_iterations"],
        writer=writer,
        base_path=config["exp_dir"],
    )
    agent.train()
    agents.save(agent, os.path.join(config["exp_dir"], "agent_final.pkl"))
    return agent

def parse_args(config_def=CONFIG_DEF):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    for config_var in config_def:
        parser.add_argument("--%s" % config_var.pop("name"), **config_var)

    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    config = parse_args()
    print(config)
    main(config)
