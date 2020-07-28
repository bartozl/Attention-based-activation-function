import torch
import torch.nn as nn
import mixed_activations

device = 'cuda' if torch.cuda.is_available() is True else 'cpu'


class Network(nn.Module):
    def __init__(self, network, nn_layers, dataset, act, combinator, norm, init, drop, hr_test=None):
        super(Network, self).__init__()

        out_neurons = 784 if dataset == 'MNIST' else 3072
        n_classes = 10 if dataset in ['MNIST', 'CIFAR10'] else None
        self.layers_list = nn.ModuleList()
        self.act_list = nn.ModuleList()
        self.nn_layers = nn_layers

        for layer in range(self.nn_layers):
            if network == 1:  # in-out neurons' number is divided by 2 for each new linear layer
                in_neurons = out_neurons
                out_neurons = in_neurons//2
            else:  # in-out neurons' number is 300 for each linear layer
                in_neurons = out_neurons
                out_neurons = 300
            if layer == self.nn_layers-1:
                self.layers_list.append(nn.Linear(in_neurons, n_classes))
            else:
                self.layers_list.append(nn.Linear(in_neurons, out_neurons))
                self.act_list.append(mixed_activations.MIX(act, combinator, out_neurons, normalize=norm,
                                                           init=init, alpha_dropout=drop, hr_test=hr_test))

        self.out = nn.LogSoftmax(dim=1)

    def forward(self, s):
        l_out, act_out = None, s
        for i in range(len(self.layers_list)):
            l_out = self.layers_list[i](act_out)
            if i < self.nn_layers - 1:
                act_out = self.act_list[i](l_out)[0]
        out = self.out(l_out)
        return out, None, None, None


class Antirelu(nn.Module):
    def __init__(self):
        super(Antirelu, self).__init__()

    def forward(self, s):
        return torch.min(torch.zeros(s.shape).to(device), s)
        # return torch.min(torch.zeros(s.shape), s)


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
