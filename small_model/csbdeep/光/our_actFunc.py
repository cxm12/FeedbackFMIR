import torch
import torch.nn as nn
import sys


class OurActFunc(nn.Module):
    def __init__(self, delta_T = 0.08, sat_I = 7.268, ns_T = 0.572):
        super(OurActFunc, self).__init__()
        self.delta_T = delta_T
        self.sat_I = sat_I
        self.ns_T = ns_T

    def act_func(self, x):
        y = x * (1 - self.delta_T * torch.exp(-x / self.sat_I) - self.ns_T)
        return y

    def forward(self, x):
        return self.act_func(x)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np

    
    delta_T = torch.arange(0, 1, 0.1)
    sat_I = torch.arange(0, 10, 1)
    ns_T = torch.arange(0, 1, 0.1)
    
    for i in delta_T:
        for k in ns_T:
            plt.figure()
            
            for j in sat_I:
                our_func = OurActFunc(delta_T=i, sat_I=j, ns_T=k)
                x = np.arange(0, 40, 1)
                y = our_func(torch.from_numpy(x))
                
                title = 'delta_T={:.2f}, sat_I={:.2f}, ns_T={:.2f}'.format(i, j, k)
                plt.plot(x, y, label=title)
                
            plt.title(title)
            # plt.ylim(0, 15)
            plt.legend()
            plt.savefig("D:\\python\\MINIST\\model\\{}.jpg".format(title))
            plt.show()
