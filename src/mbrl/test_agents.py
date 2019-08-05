import os
import torch
from tensorboardX import SummaryWriter

from src.mbrl.logger import logger
from src.mbrl.agents import MPCAgent
from src.mbrl.env_wrappers import EnvWrapper
from src.mbrl.planners import GradientDescentPlanner
from src.mbrl.models import Model, CoshLoss, SmoothAbsLoss

exp_dir = 'test_agents_exp'
logger.setup(exp_dir, os.path.join(exp_dir, 'log.txt'), 'debug')

environment = EnvWrapper.load('reacher', 'easy', visualize_reward=True)
planner = GradientDescentPlanner
model = Model(environment.state_dim, environment.action_dim)

l2_penalty = 0  # 0.001
learn_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=l2_penalty)

action_cost = CoshLoss()

goal_state = environment.set_goal()
state_cost = SmoothAbsLoss(weights=environment.get_goal_weights(), goal_state=goal_state)
writer = SummaryWriter(log_dir=exp_dir)

horizon = 20
rollout_length = 5
num_rollouts_per_iteration = 5
num_train_iterations = 5

agent = MPCAgent(
    environment,
    planner,
    model,
    horizon,
    action_cost,
    state_cost,
    optimizer,
    rollout_length,
    num_rollouts_per_iteration,
    num_train_iterations,
    writer=writer,
)

agent.train()

