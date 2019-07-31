import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.mbrl.data import TransitionsDataset, TransitionsSampler

class DynamicsModel(nn.Module):
    def forward(self, x, unnormalize=None):
        out = self._forward(x)
        if unnormalize:
            out = unnormalize(out)

        return out
    def train_model(
        self, dataset: TransitionsDataset, optimizer: torch.optim.Optimizer, batch_size: int, num_epochs: int
    ):
        train_data = DataLoader(dataset, batch_size=batch_size, sampler=TransitionsSampler(dataset))
        for epoch in range(num_epochs):
            for trans in train_data:
                loss = torch.tensor(0.0, requires_grad=True)
                state = trans[0]
                for j in range(0, len(trans) - 3, 3):
                    # state = trans[j]
                    action = trans[j + 1]
                    next_state = trans[j + 3]

                    state_action = torch.cat([state, action], 1)
                    next_state_hat = self.forward(state_action)
                    if j == 0:
                        single_loss = self.criterion(next_state_hat, next_state)

                    loss += self.criterion(next_state_hat, next_state)
                    # loss = self.criterion(next_state_hat, next_state) ### What if I only train on the last one?
                    state = next_state_hat  # Use predicted next state in next iteration

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

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
            torch.sqrt((x * self.weights) ** 2 + self.alpha ** 2) - self.alpha
        )


class CoshLoss(nn.Module):
    """
    alpha is a control limiting parameter, to avoid infeasible actions.
    """

    def __init__(self, alpha=0.25):
        super(CoshLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return (self.alpha ** 2) * torch.mean(torch.cosh(x / self.alpha) - 1)


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

