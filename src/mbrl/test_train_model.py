from src.mbrl.env_wrappers import EnvWrapper
environment = EnvWrapper.load('reacher', 'easy', visualize_reward=True)

from src.mbrl.data import TransitionsDataset
dataset = TransitionsDataset(rollouts=[], transitions_capacity=100000)

from src.mbrl.models import Model
from tensorboardX import SummaryWriter

model = Model(environment.state_dim, environment.action_dim)
model.writer = SummaryWriter(log_dir='model-test')

from torch.optim import SGD
optimizer = SGD(model.parameters(), lr=0.1)

num_rollouts = 100
rollout_length = 500
rollouts = [
    environment.get_rollout(rollout_length)
    for i in range(num_rollouts)
]
dataset.add_rollouts(rollouts)

import numpy as np
evals = model.evaluate_model(dataset=dataset, batch_size=500)
before_train_loss = np.mean(evals)
print('Before train loss:', before_train_loss)

model.train_model(dataset=dataset, optimizer=optimizer, batch_size=500, num_epochs=2)

evals = model.evaluate_model(dataset=dataset, batch_size=50)
after_train_loss = np.mean(evals)
print('After train loss:', after_train_loss)
assert before_train_loss > after_train_loss