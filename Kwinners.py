import torch
from torch import nn


class Kwinners(nn.Module):
    def __init__(self, neurons, k):
        super(Kwinners, self).__init__()
        self.neurons = neurons
        self.k = k

    def forward(self, s):
        neurons = self.neurons
        k = self.k
        idx = torch.sort(s)[1]
        for i in range(len(s)):
            s[i][idx[i][:neurons-k]] = 0
        return s
