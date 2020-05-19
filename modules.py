import torch
import torch.nn as nn


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

    def forward(self, x):
        x = self.mlp(x)
        return x


class MLP_ATT(nn.Module):
    def __init__(self, combinator):
        super(MLP_ATT, self).__init__()

        if combinator in ['MLP_ATT', 'MLP_ATT_b']:  # 105738 parameters
            self.mlp = torch.nn.Sequential(nn.Linear(4, 3),
                                           nn.ReLU(),
                                           nn.Dropout(0.2),
                                           nn.Linear(3, 4),
                                           )
        if combinator in ['MLP_ATT_neg']:  # 105738 parameters
            self.mlp = torch.nn.Sequential(nn.Linear(8, 5),
                                           nn.ReLU(),
                                           nn.Dropout(0.2),
                                           nn.Linear(5, 8),
                                           )

        if combinator == 'MLP_ATT2':  # 106890 parameters
            self.mlp = torch.nn.Sequential(nn.Linear(4, 4),
                                           nn.ReLU(),
                                           nn.Dropout(0.2),
                                           nn.Linear(4, 4),
                                           )

    def forward(self, x):
        x = self.mlp(x)
        return x
