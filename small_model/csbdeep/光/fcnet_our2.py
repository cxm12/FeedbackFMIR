import torch
import torch.nn as nn
import torch.nn.functional as F
from model.our_actFunc import OurActFunc


class fcnet_our2(nn.Module):
    def __init__(self):
        super(fcnet_our2, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.our_func = OurActFunc(delta_T=0.3, sat_I=8.116, ns_T=0.532)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.our_func(x)
        x = self.fc2(x)
        x = self.our_func(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
