import torch
import torch.nn as nn


class Model(nn.Module):
     
    def __init__(self, state_dim, action_dim, hidden_units=50, noise=None):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, state_dim)

        self.activation_fn = nn.ReLU()
        self.noise = noise
        
    def forward(self, x):
        x = self.activation_fn(self.linear1(x)) 
        x = self.activation_fn(self.linear2(x))
        x = self.linear3(x)

        return x if self.noise is None else x + torch.randn_like(x) * self.noise


class LinearModel(nn.Module):
     
    def __init__(self, state_dim, action_dim, noise=None):
        super(LinearModel, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, state_dim)
        self.noise = noise
        
    def forward(self, x):
        x = self.linear1(x)

        return x if self.noise is None else x + torch.randn_like(x) * self.noise


class ModelWithReward(nn.Module):
     
    def __init__(self, state_dim, action_dim, hidden_units=200):
        super(ModelWithReward, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, state_dim)
        self.linear4 = nn.Linear(hidden_units, 1)
        self.activation_fn = nn.ReLU()
        
    def forward(self, state, action):
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
        x = (x - self.goalState)
        return torch.sum(torch.sqrt((x*self.weights)**2 + self.alpha**2) - self.alpha)


class CoshLoss(nn.Module):
    """
    alpha is a control limiting parameter, to avoid infeasible actions.
    """
    def __init__(self, alpha=0.25):
        super(CoshLoss, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return (self.alpha**2)*torch.mean(torch.cosh(x/self.alpha) - 1)


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



