import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    """Return interval for weight initialization of hidden layers."""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model.

    This class represents the actor policy model.

    Class attributes
    ----------------
    seed : int
        Random number seed for pytorch
    fc1 : nn.Linear
        First linear hidden layer
    fc2 : nn.Linear
        Second linear hidden layer
    fc3 : nn.Linear
        Linear output layer
    bn : nn.BatchNorm1d
        One dimensional batch normalization layer

    Methods
    -------
    reset_parameters()
        Resets parameters of all linear layers.
    forward(state)
        Returns the action suggested by the actor policy model for a state.
    """
    def __init__(self, state_size, action_size, seed, 
                 fc1_units=128, fc2_units=128):
        """Build actor model and initialize parameters."""
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Resets parameters of all linear layers."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Maps states -> actions."""
        x = F.relu(self.bn(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model.

    This class represents the critic policy model.

    Class attributes
    ----------------
    seed : int
        Random number seed for pytorch
    fc1 : nn.Linear
        First linear hidden layer
    fc2 : nn.Linear
        Second linear hidden layer
    fc3 : nn.Linear
        Linear output layer
    bn : nn.BatchNorm1d
        One dimensional batch normalization layer

    Methods
    -------
    reset_parameters()
        Resets parameters of all linear layers.
    forward(state, action)
        Returns a Q-value for the provided state/action pair.
    """

    def __init__(self, state_size, action_size, seed, 
                 fc1_units=128, fc2_units=128):
        """Build actor model and initialize parameters."""
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Resets parameters of all linear layers."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Maps (state, action) pairs -> Q-values."""
        x = F.relu(self.bn(self.fc1(state)))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
