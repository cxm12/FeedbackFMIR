import torch
import torch.nn as nn
import torch.nn.functional as F
from model.network.our_actFunc import OurActFunc


class fcnet_our4(nn.Module):
    def __init__(self):
        super(fcnet_our4, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.our_func = OurActFunc(delta_T=0.237, sat_I=9.527, ns_T=0.963)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.our_func(x)
        x = self.fc2(x)
        x = self.our_func(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
