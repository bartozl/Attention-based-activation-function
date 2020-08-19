import torch
from torch import nn

from modules import MLP, Antirelu, Identity

MLP_neg = ['MLP1_neg', 'MLP_ATT_neg']
ATT_list = ['MLP_ATT', 'MLP_ATT_neg', 'MLP_ATT_b']
MLP_list = ['MLP1', 'MLP1_neg', 'MLP2', 'MLP3', 'MLP4', 'MLP5']


class MIX(nn.Module):
    def __init__(self, act_fn, combinator, neurons, normalize=None, init='random',
                 alpha_dropout=None, hr_test=None):
        super(MIX, self).__init__()
        self.combinator = combinator  # name of the combinator, e.g. "Linear"
        self.act_fn = act_fn  # basic activation function to be used, e.g. "Tanh, Sigmoid"
        self.normalize = normalize  # normalize alpha, e.g. with a Sigmoid
        self.neurons = neurons  # number of neurons of the layer
        self.alpha_dropout = alpha_dropout  # apply a dropout on alpha (only for MLP_ATT)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.act_module = {'relu': nn.ReLU(),  # dictionary containing useful functions
                           'sigmoid': nn.Sigmoid(),
                           'tanh': nn.Tanh(),
                           'antirelu': Antirelu(),
                           'identity': Identity(),
                           'softmax': nn.Softmax(dim=-1)}
        self.hr_test = hr_test
        # TODO: assert hr_test != False implies combinator=='MLP_ATT_b'
        if combinator == 'Linear':  # 3 different alpha initialization for the Linear combinator
            assert init in ['normal', 'uniform', 'random'], "init must be 'normal','uniform','random'"
            if init == 'normal':  # sample taken from a gaussian N(0,1)
                self.alpha = nn.Parameter(torch.randn(neurons, len(act_fn)), requires_grad=True)
            elif init == 'uniform':  # same init for each alpha, equal to 1/(num of act_fn)
                self.alpha = nn.Parameter(torch.ones(neurons, len(act_fn)) / len(act_fn), requires_grad=True)
            elif init == 'random':  # sample alpha in a uniform interval
                self.alpha = nn.Parameter(torch.FloatTensor(neurons, len(act_fn)).uniform_(-0.5, 0.5),
                                          requires_grad=True)

        elif combinator in MLP_list + ATT_list:  # create a list of MLP
            self.MLP_list = nn.ModuleList([MLP(combinator).to(self.device) for _ in range(neurons)]).to(self.device)
            if combinator == 'MLP_ATT_b':
                self.beta = nn.Parameter(torch.FloatTensor(neurons).uniform_(-0.5, 0.5),
                                         requires_grad=True).to(self.device)

        elif combinator == 'MLPr':  # MLPr is a mix of MLP1, MLP2
            self.MLP_list = nn.ModuleList([])
            for i in range(neurons // 2):
                self.MLP_list.extend([MLP('MLP1')])
                self.MLP_list.extend([MLP('MLP2')])

    def forward(self, s):
        act_fn = self.act_fn
        combinator = self.combinator
        alpha_dropout = self.alpha_dropout
        normalize = self.normalize
        act_module = self.act_module
        res, alpha, beta, params = None, None, None, None

        if combinator != 'None':

            if combinator not in MLP_neg:  # compute basic activations results, e.g. [tanh(s), sigmoid(s)] w/ s = input
                activations = torch.cat([act_module[act](s).unsqueeze(-1) for act in act_fn], dim=-1)

            else:  # for MLP_neg also the negative basic activations are added in the list
                temp_act = []
                for act in act_fn:
                    a = act_module[act](s).unsqueeze(-1)
                    temp_act.append(a)
                    temp_act.append(-1 * a)
                activations = torch.cat(temp_act, dim=-1)

            if combinator == 'Linear':
                # the result is the linear combination of the basic activations, weighted by alpha (learned by the nn)
                alpha = self.alpha
                if normalize != 'None':  # apply normalization if requested
                    alpha = act_module[normalize.lower()](alpha)
                res = torch.sum(alpha * activations, axis=-1)

            elif combinator in ATT_list:
                # the result is the linear combination of the basic activations, weighted by alpha (learned by an MLP)
                # each neuron is associated with a MLP with (input, output) = (n, n) where n = num. of basic act_fn

                alpha = torch.cat([self.act_module['softmax'](mod(activations[:, i, :])).unsqueeze(1)
                                   for i, mod in enumerate(self.MLP_list)], dim=1)  # e.g. [256, 128, 4]
                # print('activations:', activations.device, 'alpha:', alpha.device)


                # params = torch.cat([x.view(-1) for mod in self.MLP_list for x in mod.parameters()], dim=-1)  # l1_pen

                if alpha_dropout is not None and alpha_dropout != 0.0:  # apply dropout if required
                    alpha = nn.Dropout(alpha_dropout)(alpha)

                if combinator == 'MLP_ATT_b':
                    beta = self.beta
                    res = torch.sum(alpha * activations, axis=-1) + beta

                else:  # combinator in ['MLP_ATT', 'MLP_ATT_neg']
                    res = torch.sum(alpha * activations, axis=-1)

                # hard routing test
                if (not self.training) and (self.hr_test is not None):
                    alpha_max, idx = torch.max(alpha, dim=2)
                    mask = torch.arange(alpha.size(-1)).reshape(1, 1, -1) == idx.unsqueeze(-1)
                    res = activations[mask].reshape(alpha_max.shape)

                    if self.hr_test != 0.0:
                        # if the computed activation is not larger enough than the others, use a default activation.
                        alpha_mid_range = torch.abs(torch.max(alpha, dim=-1, keepdim=True)[0] -
                                                    torch.min(alpha, dim=-1, keepdim=True)[0]) / 2
                        res = res.unsqueeze(-1)
                        # print(torch.max(alpha, dim=-1)[0][0][0], torch.min(alpha, dim=-1)[0][0][0])
                        condition = alpha_mid_range >= self.hr_test
                        if self.combinator == 'MLP_ATT_neg':
                            act_by_inspection = torch.zeros(s.shape).unsqueeze(-1)
                        elif self.act_fn == ['antirelu', 'identity', 'relu', 'sigmoid']:
                            act_by_inspection = s.unsqueeze(-1)
                        else:
                            act_by_inspection = activations[:, :, -1].unsqueeze(-1)  # tanh activation
                        res = torch.where(condition, res, act_by_inspection).squeeze(-1)
            else:  # combinator in ['MLP1', 'MLP2', 'MLP3', 'MLP4', 'MLP5', 'MLPr', 'MLP1_neg]
                # the results will be computed by an MLP with dim (input, output) = (n,1) where n = num. of act_fn
                res = torch.cat([mod(activations[:, i, :]) for i, mod in enumerate(self.MLP_list)], dim=-1)

        else:  # compute only a basic activation function (no MIX)
            res = act_module[act_fn[0]](s)
        return res, alpha, beta, params


class MIX_jit(nn.Module):

    def __init__(self, act_fn, combinator, neurons, normalize=None, init='random',
                 alpha_dropout=None, hr_test=None):
        super(MIX_jit, self).__init__()
        self.combinator = combinator  # name of the combinator, e.g. "Linear"
        self.act_fn = act_fn  # basic activation function to be used, e.g. "Tanh, Sigmoid"
        self.normalize = normalize  # normalize alpha, e.g. with a Sigmoid
        self.neurons = neurons  # number of neurons of the layer
        self.alpha_dropout = alpha_dropout  # apply a dropout on alpha (only for MLP_ATT)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.act_module = torch.nn.ModuleDict({'relu': nn.ReLU(),  # dictionary containing useful functions
                                               'sigmoid': nn.Sigmoid(),
                                               'tanh': nn.Tanh(),
                                               'antirelu': Antirelu(),
                                               'identity': Identity(),
                                               'softmax': nn.Softmax(dim=-1)})
        self.hr_test = hr_test

        # for JIT, they must be initialized
        self.res = torch.tensor(0)
        self.alpha = torch.tensor(0)
        self.beta = torch.tensor(0)
        self.params = torch.tensor(0)

        if combinator == 'Linear':  # 3 different alpha initialization for the Linear combinator
            assert init in ['normal', 'uniform', 'random'], "init must be 'normal','uniform','random'"
            if init == 'normal':  # sample taken from a gaussian N(0,1)
                self.alpha = nn.Parameter(torch.randn(neurons, len(act_fn)), requires_grad=True)
            elif init == 'uniform':  # same init for each alpha, equal to 1/(num of act_fn)
                self.alpha = nn.Parameter(torch.ones(neurons, len(act_fn)) / len(act_fn), requires_grad=True)
            elif init == 'random':  # sample alpha in a uniform interval
                self.alpha = nn.Parameter(torch.FloatTensor(neurons, len(act_fn)).uniform_(-0.5, 0.5),
                                          requires_grad=True)

        elif combinator in MLP_list + ATT_list:  # create a list of MLP
            self.MLP_list = nn.ModuleList([MLP(combinator) for _ in range(neurons)])
            if combinator == 'MLP_ATT_b':
                self.beta = nn.Parameter(torch.FloatTensor(neurons).uniform_(-0.5, 0.5), requires_grad=True)

        elif combinator == 'MLPr':  # MLPr is a mix of MLP1, MLP2
            self.MLP_list = nn.ModuleList([])
            for i in range(neurons // 2):
                self.MLP_list.extend([MLP('MLP1')])
                self.MLP_list.extend([MLP('MLP2')])

    def forward(self, s):
        act_fn = self.act_fn
        combinator = self.combinator
        alpha_dropout = self.alpha_dropout
        normalize = self.normalize
        act_module = self.act_module
        res = self.res
        alpha = self.alpha
        beta = self.beta
        params = self.params

        if combinator != 'None':

            if combinator != 'MLP1_neg' and combinator != 'MLP_ATT_neg':  # compute basic activations results, e.g. [tanh(s), sigmoid(s)] w/ s = input
                activations = []
                for key, act_mod in act_module.items():
                    for act in act_fn:
                        if act == key:
                            activations.append(act_mod(s).unsqueeze(-1))
                activations = torch.cat(activations, dim=-1)
            else:  # for MLP_neg also the negative basic activations are added in the list
                temp_act = []
                for key, act_mod in act_module.items():
                    for act in act_fn:
                        if act == key:
                            a = act_mod(s).unsqueeze(-1)
                            temp_act.append(a)
                            temp_act.append(-1 * a)
                activations = torch.cat(temp_act, dim=-1)

            if combinator == 'Linear':
                # the result is the linear combination of the basic activations, weighted by alpha (learned by the nn)
                if normalize != 'None':  # apply normalization if requested
                    for key, act_mod in act_module.items():
                        if key == normalize.lower():
                            alpha = act_mod(alpha)
                    # alpha = act_module[normalize.lower()](alpha)
                res = torch.sum(alpha * activations, dim=-1)

            elif combinator == 'MLP_ATT_neg' or combinator == 'MLP_ATT' or combinator == 'MLP_ATT_b':
                # the result is the linear combination of the basic activations, weighted by alpha (learned by an MLP)
                # each neuron is associated with a MLP with (input, output) = (n, n) where n = num. of basic act_fn
                alpha_temp = []
                for i, mod in enumerate(self.MLP_list):
                    for act, act_mod in act_module.items():
                        if act == 'softmax':
                            alpha_temp.append(act_mod(mod(activations[:, i, :])).unsqueeze(1))
                alpha = torch.cat(alpha_temp, dim=1)

                # alpha = torch.cat([self.act_module['softmax'](mod(activations[:, i, :])).unsqueeze(1)
                #                   for i, mod in enumerate(self.MLP_list)], dim=1)  # e.g. [256, 128, 4]

                # params = torch.cat([x.view(-1) for mod in self.MLP_list for x in mod.parameters()], dim=-1)  # l1_pen

                if alpha_dropout is not None and alpha_dropout != 0.0:  # apply dropout if required
                    alpha = nn.functional.dropout(alpha, p=alpha_dropout)

                if combinator == 'MLP_ATT_b':
                    res = torch.sum(alpha * activations, dim=-1) + beta

                else:  # combinator in ['MLP_ATT', 'MLP_ATT_neg']
                    res = torch.sum(alpha * activations, dim=-1)

                # hard routing test
                '''
                if (not self.training) and (self.hr_test is not None):
                    alpha_max, idx = torch.max(alpha, dim=2)
                    mask = torch.arange(alpha.size(-1)).reshape(1, 1, -1) == idx.unsqueeze(-1)
                    res = activations[mask].reshape(alpha_max.shape)

                    if self.hr_test != 0.0:
                        # if the computed activation is not larger enough than the others, use a default activation.
                        alpha_mid_range = torch.abs(torch.max(alpha, dim=-1, keepdim=True)[0] -
                                                    torch.min(alpha, dim=-1, keepdim=True)[0])/2
                        res = res.unsqueeze(-1)
                        # print(torch.max(alpha, dim=-1)[0][0][0], torch.min(alpha, dim=-1)[0][0][0])
                        condition = alpha_mid_range >= self.hr_test
                        if self.combinator == 'MLP_ATT_neg':
                            act_by_inspection = torch.zeros(s.shape).unsqueeze(-1)
                        elif self.act_fn == ['antirelu', 'identity', 'relu', 'sigmoid']:
                            act_by_inspection = s.unsqueeze(-1)
                        else:
                            act_by_inspection = activations[:, :, -1].unsqueeze(-1)  # tanh activation
                        res = torch.where(condition, res, act_by_inspection).squeeze(-1)
                '''
            else:  # combinator in ['MLP1', 'MLP2', 'MLP3', 'MLP4', 'MLP5', 'MLPr', 'MLP1_neg]
                # the results will be computed by an MLP with dim (input, output) = (n,1) where n = num. of act_fn
                res_temp = []
                for i, mod in enumerate(self.MLP_list):
                    res_temp.append(mod(activations[:, i, :]))
                res = torch.cat(res_temp, dim=-1)

        else:  # compute only a basic activation function (no MIX)
            for act, act_mod in act_module.items():
                if act == act_fn[0]:
                    res = act_mod(s)

        return res
