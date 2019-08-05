from src.mbrl.logger import logger
import os
exp_dir = 'test_agents_derp'
logger.setup(exp_dir, os.path.join(exp_dir, 'log.txt'), 'debug')

from src.mbrl.agents import *
environment = EnvWrapper.load('reacher', 'easy', visualize_reward=True)

from src.mbrl.planners import GradientDescentPlanner
planner = GradientDescentPlanner
from src.mbrl.models import Model, CoshLoss, SmoothAbsLoss
from tensorboardX import SummaryWriter

model = Model(environment.state_dim, environment.action_dim)

l2_penalty = 0  # 0.001
learn_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=l2_penalty)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

action_cost = CoshLoss()

import numpy as np
goal_state = environment.set_goal()
state_cost = SmoothAbsLoss(weights=environment.get_goal_weights(), goal_state=goal_state)

horizon = 5
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

