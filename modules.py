import torch
import torch.nn as nn
import mixed_activations


class Network(nn.Module):
    def __init__(self, act, combinator, norm, init, drop):
        super(Network, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.mix = mixed_activations.MIX(act, combinator, neurons=128, normalize=norm, init=init, alpha_dropout=drop)
        self.l2 = nn.Linear(128, 10)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, s):
        l1_out = self.l1(s)
        mix_out, alpha, beta, params = self.mix(l1_out)
        l2_out = self.out(self.l2(mix_out))
        return l2_out, alpha, beta, params


class Antirelu(nn.Module):
    def __init__(self):
        super(Antirelu, self).__init__()

    def forward(self, s):
        return torch.min(torch.zeros(s.shape), s)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, s):
        return s


class MLP(nn.Module):
    def __init__(self, combinator):
        super(MLP, self).__init__()

        if combinator == 'MLP1':  # 104202 parameters
            self.mlp = torch.nn.Sequential(nn.Linear(4, 3),
                                           nn.ReLU(),
                                           nn.Dropout(0.2),
                                           nn.Linear(3, 1),
                                           )
        if combinator == 'MLP1_neg':  # 104202 parameters
            self.mlp = torch.nn.Sequential(nn.Linear(8, 5),
                                           nn.ReLU(),
                                           nn.Dropout(0.2),
                                           nn.Linear(5, 1),
                                           )

        if combinator == 'MLP2':  # 104970
            self.mlp = torch.nn.Sequential(nn.Linear(4, 4),
                                           nn.ReLU(),
                                           nn.Dropout(0.2),
                                           nn.Linear(4, 1),
                                           )

        if combinator == 'MLP3':  # 104202 parameters --> same of MLP1 but w/out dropout
            self.mlp = torch.nn.Sequential(nn.Linear(4, 3),
                                           nn.ReLU(),
                                           nn.Linear(3, 1),
                                           )

        if combinator == 'MLP4':  # 104970 --> same of MLP1 but w/out dropout
            self.mlp = torch.nn.Sequential(nn.Linear(4, 4),
                                           nn.ReLU(),
                                           nn.Linear(4, 1),
                                           )
        if combinator == 'MLP5':  # 105098
            self.mlp = torch.nn.Sequential(nn.Linear(4, 3),
                                           nn.ReLU(),
                                           nn.Dropout(0.2),
                                           nn.Linear(3, 2),
                                           nn.ReLU(),
                                           nn.Linear(2, 1),
                                           )

        if combinator in ['MLP_ATT', 'MLP_ATT_b']:  # 105738 parameters
            self.mlp = torch.nn.Sequential(nn.Linear(4, 3),
                                           nn.ReLU(),
                                           nn.Dropout(0.2),
                                           nn.Linear(3, 4),
                                           )
            #4*3+3 + 3*4 = 12+3+12
        if combinator in ['MLP_ATT_neg']:  # 105738 parameters
            self.mlp = torch.nn.Sequential(nn.Linear(8, 5),
                                           nn.ReLU(),
                                           nn.Dropout(0.2),
                                           nn.Linear(5, 8),
                                           )

    def forward(self, x):
        x = self.mlp(x)
        return x
