import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.mbrl.data import TransitionsDataset, TransitionsSampler
from src.mbrl.logger import logger

class DynamicsModel(nn.Module):
    def forward(self, x, unnormalize=None):
        out = self._forward(x)
        if unnormalize:
            out = unnormalize(out)

        return out
    def evaluate_model(self, dataset, batch_size, criterion=None):
        if not criterion:
            criterion = torch.nn.MSELoss()
        eval_data = DataLoader(dataset, batch_size=batch_size, sampler=TransitionsSampler(dataset))
        evals = []
        for inputs, outputs in eval_data:
            for ((states, observations, actions), (rewards, next_states, next_observations)) in zip(inputs, outputs):
                state_action = torch.cat([states, actions], 1)
                next_states_hat = self.forward(state_action)
                evals.append(criterion(next_states_hat, next_states).detach().numpy())
        return evals

    def train_model(
        self, dataset: TransitionsDataset, optimizer: torch.optim.Optimizer, batch_size: int, num_epochs: int, criterion=None,
    ):
        if not criterion:
            criterion = torch.nn.MSELoss()
        train_data = DataLoader(dataset, batch_size=batch_size, sampler=TransitionsSampler(dataset))
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_iters = 0
            for inputs, outputs in train_data:
                loss = 0
                for ((states, observations, actions), (rewards, next_states, next_observations)) in zip(inputs, outputs):
                    state_action = torch.cat([states, actions], 1)
                    next_states_hat = self.forward(state_action)
                    loss = loss + criterion(next_states_hat, next_states)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                epoch_loss += loss
                num_iters += 1

            epoch_loss = float(epoch_loss.detach().numpy())
            if epoch == 0:
                logger.record_tabular("LossFirstEpoch", epoch_loss/num_iters)
                logger.record_tabular("NumBatchesPerEpoch", num_iters)
        logger.record_tabular("LossLastEpoch", epoch_loss/num_iters)

            # if self.writer is not None:
            #     self.writer.add_scalar('loss/state/{}'.format(iteration), loss, epoch)
            #     self.writer.add_scalar('single_step_loss/{}'.format(iteration), single_loss, epoch)


class Model(DynamicsModel):
    def __init__(self, state_dim, action_dim, hidden_units=50, noise=None):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, state_dim)

        self.activation_fn = nn.ReLU()
        self.noise = noise

    def _forward(self, x):
        x = self.activation_fn(self.linear1(x))
        x = self.activation_fn(self.linear2(x))
        x = self.linear3(x)
        return x if self.noise is None else x + torch.randn_like(x) * self.noise



class LinearModel(DynamicsModel):
    def __init__(self, state_dim, action_dim, noise=None):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, state_dim)
        self.noise = noise

    def _forward(self, x):
        x = self.linear1(x)

        return x if self.noise is None else x + torch.randn_like(x) * self.noise


class ModelWithReward(DynamicsModel):
    def __init__(self, state_dim, action_dim, hidden_units=200):
        super(ModelWithReward, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, state_dim)
        self.linear4 = nn.Linear(hidden_units, 1)
        self.activation_fn = nn.ReLU()

    def _forward(self, state, action):
        x = torch.cat((state, action), -1)
        x = self.activation_fn(self.linear1(x))
        x = self.activation_fn(self.linear2(x))
        s_diff = self.linear3(x)
        reward = self.linear4(x)

        return s_diff, reward


class CostModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=70):
        super(CostModel, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, 1)
        self.activation_fn = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat((state, action), -1)
        x = self.activation_fn(self.linear1(x))
        x = self.activation_fn(self.linear2(x))
        cost = self.linear3(x)  # In DM Control, rewards are in the interval [0,1]

        return cost


class SmoothAbsLoss(nn.Module):
    """
    Alpha here interpolates between l_1 and l_2 norm. Larger alpha (e.g. 2) becomes more quadratic.
    """

    def __init__(self, weights, goal_state, alpha=0.4):
        super(SmoothAbsLoss, self).__init__()
        self.alpha = alpha
        self.weights = weights
        self.goalState = goal_state

    def forward(self, x):
        x = x - self.goalState
        return torch.sum(
            torch.sqrt((x * self.weights) ** 2 + self.alpha ** 2) - self.alpha,
            dim=-1
        )


class CoshLoss(nn.Module):
    """
    alpha is a control limiting parameter, to avoid infeasible actions.
    """

    def __init__(self, alpha=0.25):
        super(CoshLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return (self.alpha ** 2) * torch.mean(torch.cosh(x / self.alpha) - 1, dim=-1)


class QuadraticCost(nn.Module):
    """
    Quadratic multiplier given linear
    """

    def __init__(self, dim, goal_state):
        super(QuadraticCost, self).__init__()
        self.linear = nn.Linear(dim, dim)
        self.goalState = goal_state

    def forward(self, x):
        a = self.linear(x - self.goalState)
        out = torch.dot(x - self.goalState, a)
        return out

