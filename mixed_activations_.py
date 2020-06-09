import torch
from torch import nn
from modules import MLP, MLP_ATT, Antirelu, Identity


# class MIX(torch.jit.ScriptModule):
# noinspection PyTypeChecker
class MIX(nn.Module):
    __constants__ = ['MLP_list', 'act_module']  # needed for JIT

    def __init__(self, act_fn, combinator, neurons, normalize=None,
                 init='random', alpha_dropout=None):
        super(MIX, self).__init__()
        self.combinator = combinator
        self.act_fn = act_fn
        self.normalize = normalize
        self.neurons = neurons
        self.alpha_dropout = alpha_dropout
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.act_module = nn.ModuleDict([['relu', nn.ReLU()],
                                         ['sigmoid', nn.Sigmoid()],
                                         ['tanh', nn.Tanh()],
                                         ['antirelu', Antirelu()],
                                         ['identity', Identity()],
                                         ['softmax', nn.Softmax()]])

        if combinator == 'Linear':
            if init == 'normal':
                self.alpha = nn.Parameter(torch.randn(neurons, len(act_fn)))
            elif init == 'uniform':
                self.alpha = nn.Parameter(torch.ones(neurons, len(act_fn)) / len(act_fn))
            elif init == 'random':
                self.alpha = nn.Parameter(torch.FloatTensor(neurons, len(act_fn)).uniform_(-0.5, 0.5))
        elif combinator in ['MLP1', 'MLP1_neg', 'MLP2', 'MLP3', 'MLP4', 'MLP5']:
            self.MLP_list = nn.ModuleList([MLP(combinator) for _ in range(neurons)])

        elif combinator == 'MLPr':  # 104586 parameters
            self.MLP_list = nn.ModuleList([])
            for i in range(neurons // 2):
                self.MLP_list.extend([MLP('MLP1')])
                self.MLP_list.extend([MLP('MLP2')])

        elif combinator in ['MLP_ATT', 'MLP_ATT_neg', 'MLP_ATT_b']:
            self.MLP_list = nn.ModuleList([MLP_ATT(combinator) for _ in range(neurons)])


    # @torch.jit.script_method
    # noinspection PyArgumentList
    def forward(self, s):
        act_fn = self.act_fn
        combinator = self.combinator
        alpha_dropout = self.alpha_dropout
        normalize = self.normalize
        res = []

        if combinator != 'None':
            # jit compatible implementation (no indexing supported)
            activations = []
            for act in act_fn:
                for key, module in self.act_module.items():
                    if key == act:
                        act_temp = module(s).unsqueeze(-1)
                        activations.append(act_temp)
                        if combinator in ['MLP_ATT_neg', 'MLP1_neg']:
                            activations.append(-act_temp)
            activations = torch.cat(activations, dim=-1)

            if combinator == 'Linear':
                alpha = self.alpha
                if normalize != 'None':
                    alpha = self.act_module[normalize.lower()](alpha)
                res = torch.sum(alpha * activations, axis=-1)

            elif combinator in ['MLP_ATT', 'MLP_ATT_neg', 'MLP_ATT_b']:
                alpha = []
                for i, mod in enumerate(self.MLP_list):
                    alpha_i = self.act_module['softmax'](mod(activations[:, i, :]))  # [256,4] or [256,8]
                    alpha.append(alpha_i.unsqueeze(1))
                alpha = torch.cat(alpha, dim=1)  # [256, 128, 4] or [256, 128, 8]
                if alpha_dropout is not None:
                    alpha = nn.Dropout(alpha_dropout)(alpha)
                res = torch.sum(alpha * activations, axis=-1)
                # uncomment for hard routing
                if self.training is False:
                    alpha_max, idx = torch.max(alpha, dim=2)
                    mask = torch.arange(alpha.size(-1)).reshape(1, 1, -1) == idx.unsqueeze(-1)
                    res = activations[mask].reshape(alpha_max.shape)

            else:  # combinator in ['MLP1', 'MLP2', 'MLP3', 'MLP4', 'MLP5', 'MLPr']
                res = []
                for i, mod in enumerate(self.MLP_list):
                    res.append(mod(activations[:, i, :]))
                res = torch.cat(res, dim=-1)
        else:
            for key, module in self.act_module.items():
                if key == act_fn[0]:
                    res = module(s)
        return res
