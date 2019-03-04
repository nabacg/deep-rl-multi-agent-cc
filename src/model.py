# Critic and Actor models

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, h_1_size = 256, h_2_size = 128):
        super(ActorQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc_1 = nn.Linear(state_size, h_1_size)
        self.fc_2 = nn.Linear(h_1_size, h_2_size)
        self.output = nn.Linear(h_2_size, action_size)
        self.reset_parameters()

    def forward(self, state):
        y = F.relu(self.fc_1(state))
        y = F.relu(self.fc_2(y))
        return F.tanh(self.output(y))
    
    def reset_parameters(self):
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        

class CriticQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, h_1_size = 256, h_2_size = 128):
        super(CriticQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc_1 = nn.Linear(state_size, h_1_size)
        self.fc_2 = nn.Linear(h_1_size + action_size, h_2_size)
        self.output = nn.Linear(h_2_size, 1)
        
    def forward(self, state, action):
        y = F.relu(self.fc_1(state))
        y = torch.cat((y, action), dim=1)
        y = F.relu(self.fc_2(y))
        return self.output(y)
        
        
    def reset_parameters(self):
        self.fc_1.weight.data.uniform_(*hidden_init(self.fc_1))
        self.fc_2.weight.data.uniform_(*hidden_init(self.fc_2))
        self.output.weight.data.uniform_(-3e-3, 3e-3)