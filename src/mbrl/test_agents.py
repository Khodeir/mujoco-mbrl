from src.mbrl.logger import logger
import os
exp_dir = 'test_agents_exp'
logger.setup(exp_dir, os.path.join(exp_dir, 'log_nn_cost.txt'), 'debug')

from src.mbrl.agents import *
environment = EnvWrapper.load('reacher', 'easy', visualize_reward=True)

from src.mbrl.planners import RandomShootingPlanner
planner = RandomShootingPlanner
from src.mbrl.models import Model, CoshLoss, SmoothAbsLoss
model = Model(environment.state_dim, environment.action_dim)

l2_penalty = 0  # 0.001
learn_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=l2_penalty)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

action_cost = CoshLoss()

import numpy as np
goal_state = torch.zeros(environment.state_dim, dtype=torch.float)
goal_state[0] = np.random.uniform(low=-np.pi, high=np.pi)
goal_state[1] = np.random.uniform(low=-2.8, high=2.8)  # Avoid infeasible goals
goal_state[-1] = 0
goal_state[-2] = 0

a = 0.12*np.cos(goal_state[1])
b = 0.12*np.sin(goal_state[1])
theta = goal_state[0] + np.arctan(b/(0.12 + a))
mag = np.sqrt((0.12 + a)**2 + b**2)
target_x = mag*np.cos(theta)
target_y = mag*np.sin(theta)
environment._env.physics.named.model.geom_pos['target', 'x'] = target_x
environment._env.physics.named.model.geom_pos['target', 'y'] = target_y

state_cost = SmoothAbsLoss(weights=environment.get_goal_weights(), goal_state=goal_state)

horizon = 20
rollout_length = 250
num_rollouts_per_iteration = 5
num_train_iterations = 100
num_epochs_per_iteration = 50
batch_size = 50

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
    num_epochs_per_iteration,
    batch_size=batch_size
)

agent.train()

