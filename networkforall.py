import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    def __init__(self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, actor=False):
        super(Network, self).__init__()

        """self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_norm.weight.data.fill_(1)
        self.input_norm.bias.data.fill_(0)"""

        self.fc1 = nn.Linear(input_dim,hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim,hidden_out_dim)
        self.fc3 = nn.Linear(hidden_out_dim,output_dim)
        # self.nonlin = f.relu #leaky_relu
        self.nonlin = f.leaky_relu #leaky_relu
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.actor = actor
        #self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        if self.actor:
            # return a vector of the force
            h1 = self.nonlin(self.fc1(x))

            h2 = self.nonlin(self.fc2(h1))  
            h3 = self.fc3(h2)
            norm = torch.norm(h3)
            
            # h3 is a 2D vector (a force that is applied to the agent)
            # we bound the norm of the vector to be between 0 and 10
            # return (torch.tanh(norm))*h3/norm if norm > 0 else h3
            # return torch.sigmoid(h3)
            # return h3
            # return self.softmax(h3)
            return self.sigmoid(h3)
        
        else:
            # critic network simply outputs a number
            h1 = self.nonlin(self.fc1(x))
            h2 = self.nonlin(self.fc2(h1))
            h3 = (self.fc3(h2))
            return h3