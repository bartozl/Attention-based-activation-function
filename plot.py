import os
import json
from collections import defaultdict
import pandas as pd
import imgkit
import matplotlib.pyplot as plt
import argparse
import torch
from torch import nn
from mixed_activations import MIX, Antirelu, Identity
import numpy as np
from pathlib import Path
import seaborn as sns

""" ADD: min accuracy after 40 epochs
    ADD: max accuracy
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
neurons = 128
MLP_LIST = ['MLP1', 'MLP1_neg', 'MLP2', 'MLP3', 'MLP4', 'MLP5', 'MLPr']
ATT_LIST = ['MLP_ATT', 'MLP_ATT_neg']
COMBINED_ACT = ['antirelu_identity_sigmoid_tanh', 'antirelu_identity_relu_sigmoid', 'identity_relu_sigmoid_tanh']
n_samples = 500
input_ = torch.Tensor(np.linspace(-2, 2, n_samples).astype(np.float32)).to(device)
input_ = input_.repeat(neurons, 1)  # input_.shape = [128,500]
act_module = nn.ModuleDict([['relu', nn.ReLU()],
                            ['sigmoid', nn.Sigmoid()],
                            ['tanh', nn.Tanh()],
                            ['antirelu', Antirelu()],
                            ['identity', Identity()],
                            ['softmax', nn.Softmax()]])
n_epochs = 20


def compute_Linear_activations(results, epoch):
    alpha = []
    for i, act in enumerate(results["act_fn"]):
        # noinspection PyArgumentList
        alpha.append(torch.Tensor(results["alpha_per_epoch"][epoch][act]).unsqueeze(-1))
    alpha = torch.cat(alpha, dim=-1).to(device)

    activations = []  # jit compatible implementation
    for a in results['act_fn']:
        for key, module in act_module.items():
            if key == a:
                activations.append(module(input_[0]).unsqueeze(0))
    activations = torch.cat(activations, dim=0).to(device)
    if results['normalize'] != 'None':
        alpha = act_module[results['normalize'].lower()](alpha)
    output = alpha @ activations
    return output


def compute_MLP_activations(results, epoch, mix, path):
    state_dict = torch.load(f'{path}/weights/{epoch}.pth')  # load the model of the original network
    state_dict_filt = {k[2:]: v for k, v in state_dict.items()}  # adjust the name of the parameters
    del state_dict_filt['weight']  # delete the parameter of the original network linear layers
    del state_dict_filt['bias']
    mix.load_state_dict(state_dict_filt)  # load the MIX parameters
    mix.eval()  # evaluation mode --> no dropout!

    activations = []  # jit compatible implementation
    for a in results['act_fn']:
        for key, module in act_module.items():
            if key == a:
                act_temp = module(input_).unsqueeze(-1)
                activations.append(act_temp)
                if results['combinator'] in ['MLP_ATT_neg', 'MLP1_neg']:
                    activations.append(-act_temp)
    activations = torch.cat(activations, dim=-1)
    # print(activations.shape)  # [128, 500, 4] = [neurons, input_, fn_act]
    mlp_output = []

    for i, mod in enumerate(mix.MLP_list):
        mlp_output.append(mod(activations[i, :, :]).unsqueeze(0))
    if results['combinator'] in MLP_LIST:
        output = torch.cat(mlp_output, dim=0).squeeze(-1)
    else:
        alpha = torch.cat(mlp_output, dim=0)  # alpha.shape = [128, 500, 4]
        alpha = nn.functional.softmax(alpha, dim=-1)
        output = torch.sum(alpha * activations, axis=-1)
    # print(results["combinator"], 'output.shape', output.shape)
    return output


def plot_activations(path_dict):
    tick_fontsize = 15
    for act in path_dict:
        for path in path_dict[act]:
            with open(path + '/results.json', 'r') as f:
                results = json.load(f)
            dest_path = f'{path}/plot/'
            if len(os.listdir(dest_path)) == 12:
                # print('path already plotted')
                continue
            print(path)
            for epoch in results['train_acc'].keys():
                if int(epoch) % n_epochs != 0 and int(epoch) != 1:
                    continue
                if results['combinator'] == 'Linear':
                    output = compute_Linear_activations(results, epoch)
                elif results['combinator'] in MLP_LIST+ATT_LIST:
                    mix = MIX(results['act_fn'],
                              combinator=results['combinator'],
                              neurons=neurons,
                              normalize=results['normalize'])
                    output = compute_MLP_activations(results, epoch, mix, path)
                else:
                    print(f"no plot method available for {results['combinator']} combinator...")
                    continue
                if int(epoch) == 1:
                    output_1 = output
                fig, ax = plt.subplots(5, 4)
                fig.set_size_inches(32, 18)
                fig.tight_layout()
                idx = 0
                for i in range(5):
                    for j in range(4):
                        plt.setp(ax[i][j].get_xticklabels(), fontsize=tick_fontsize)
                        plt.setp(ax[i][j].get_yticklabels(), fontsize=tick_fontsize)
                        ax[i][j].set_ylim([-2, 2])
                        ax[i][j].axhline(ls='--', color='lightgray')
                        ax[i][j].axvline(ls='--', color='lightgray')
                        ax[i][j].set_title(f'[n={idx}]')
                        ax[i][j].plot(input_[idx].detach().numpy(), output[idx].detach().numpy(), color='green')
                        ax[i][j].plot(input_[idx].detach().numpy(), output_1[idx].detach().numpy(), color='red', ls='--')
                        idx += 1
                plt.savefig(f'{dest_path}/{epoch}.png',
                            bbox_inches='tight', dpi=200)
                plt.close()

                if int(epoch) == 200:
                    fig, ax = plt.subplots(11, 11)
                    fig.set_size_inches(32, 18)
                    fig.tight_layout()
                    idx = 0
                    for i in range(11):
                        for j in range(11):
                            plt.setp(ax[i][j].get_xticklabels(), fontsize=tick_fontsize)
                            plt.setp(ax[i][j].get_yticklabels(), fontsize=tick_fontsize)
                            ax[i][j].set_ylim([-2, 2])
                            ax[i][j].axhline(ls='--', color='lightgray')
                            ax[i][j].axvline(ls='--', color='lightgray')
                            ax[i][j].set_title(f'[n={idx}]')
                            ax[i][j].plot(input_[idx].detach().numpy(), output[idx].detach().numpy(), color='green')
                            ax[i][j].plot(input_[idx].detach().numpy(), output_1[idx].detach().numpy(), color='red', ls='--')
                            idx += 1
                    plt.savefig(f'{dest_path}/{epoch}_tot.png',
                                bbox_inches='tight', dpi=200)
                    plt.close()


def plot_accuracy(path_dict, dest_path):
    path_dict_ = path_dict.copy()
    title_fontsize = 45
    label_fontsize = 35
    tick_fontsize = 30
    legend_fontsize = 25
    alpha = 1
    lw = 3

    pairs_to_compare = [['MLP1', 'L_unif_none']]  # it must be a list of lists!

    for relu_path in path_dict_['relu']:
        with open(relu_path + '/results.json', 'r') as f:
            results_relu = json.load(f)
        if results_relu['batch_size'] == 256:
            break
    relu_label = 'relu'

    del path_dict_['relu']
    del path_dict_['sigmoid']
    del path_dict_['antirelu']
    del path_dict_['tanh']

    for pair in pairs_to_compare:
        for act in path_dict_.keys():
            Path(f'{dest_path}/{act}/').mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(2, 2)
            fig.tight_layout()
            fig.set_size_inches(32, 18)

            for i in range(2):
                ax[0][i].set_title('Train accuracy', fontsize=title_fontsize)
                ax[1][i].set_title('Test accuracy', fontsize=title_fontsize)

                ax[0][i].set_xlabel('epochs', fontsize=label_fontsize)
                ax[1][i].set_xlabel('epochs', fontsize=label_fontsize)

                ax[0][i].set_ylabel('accuracy [%]', fontsize=label_fontsize)
                ax[1][i].set_ylabel('accuracy [%]', fontsize=label_fontsize)

                ax[0][i].set_ylim(bottom=80, top=102)
                ax[1][i].set_ylim(bottom=80, top=98)

                plt.setp(ax[0][i].get_xticklabels(), fontsize=tick_fontsize)
                plt.setp(ax[1][i].get_xticklabels(), fontsize=tick_fontsize)
                plt.setp(ax[0][i].get_yticklabels(), fontsize=tick_fontsize)
                plt.setp(ax[1][i].get_yticklabels(), fontsize=tick_fontsize)

                ax[0][i].plot(list(results_relu['train_acc'].keys()), list(results_relu['train_acc'].values()),
                              color="blue",
                              label=relu_label, alpha=1, linewidth=lw)
                ax[1][i].plot(list(results_relu['test_acc'].keys()), list(results_relu['test_acc'].values()),
                              label=relu_label,
                              color="blue", alpha=alpha, linewidth=lw)

                for j, label_ax in enumerate(ax[0][i].xaxis.get_ticklabels()):
                    if (j + 1) % n_epochs != 0:
                        label_ax.set_visible(False)
                for j, label_ax in enumerate(ax[1][i].xaxis.get_ticklabels()):
                    if (j + 1) % n_epochs != 0:
                        label_ax.set_visible(False)

            i = 0
            for path in sorted(path_dict_[act]):
                with open(path + '/results.json', 'r') as f:
                    results = json.load(f)
                if results['run_name'] not in pair:
                    continue

                label = results['run_name']
                # print(results['run_name'])
                color = 'green' if results['run_name'] == pair[0] else 'red'
                ax[0][i].plot(list(results['train_acc'].keys()), list(results['train_acc'].values()), color=color,
                              label=label, alpha=alpha, linewidth=lw)
                ax[1][i].plot(list(results['test_acc'].keys()), list(results['test_acc'].values()), color=color,
                              label=label, alpha=alpha, linewidth=lw)
                ax[0][i].legend(fontsize=legend_fontsize, loc='center right')
                ax[1][i].legend(fontsize=legend_fontsize, loc='center right')
                i += 1
                print(pairs_to_compare)

            plt.savefig(dest_path + '/{}/{}_vs_{}.png'.format(act, pair[0], pair[1]))
            plt.close()


def plot_table(path_dict, save_path):
    # pd.set_option('display.column_space', 20)
    pd.set_option('display.precision', 4)
    pd.set_option('display.width', 40)

    for i, act in enumerate(path_dict.keys()):
        row_labels, values_train, values_test = [], [], []
        for path in path_dict[act]:
            with open(path + '/results.json', 'r') as f:
                results = json.load(f)
                if i == 0:
                    col_labels = fill_col_labels(results)
                temp_train, temp_test = fill_row_values(results, path, act)
                values_train.append(temp_train)
                values_test.append(temp_test)

        # create table
        table_train = create_table(values_train, col_labels, act, 'train',
                                   sort_by=['combinator'])
        table_test = create_table(values_test, col_labels, act, 'test', sort_by=['combinator'])

        # save table
        save_table(table_train, table_test, save_path, act)


def plot_table_max(path_dict, save_path, limit):
    res_json = ['results.json', 'results_hr.json']
    row_labels, values_train, values_test = [], [], []
    for i, act in enumerate(path_dict.keys()):
        for path in path_dict[act]:
            for res in res_json:
                try:
                    with open(f'{path}/{res}', 'r') as f:
                        results = json.load(f)
                        # att = 2 if res == 'results_hr.json' else 0
                except Exception as e:
                    continue
            # with open(path + '/results.json', 'r') as f:
            #     results = json.load(f)
                if i == 0:
                    col_labels = fill_col_labels(results, max_=True, att=2)
                temp_train, temp_test = fill_row_values(results, path, act, max_=True, att=2)
                if True not in np.where(temp_test[8] >= limit, True, False):
                    continue
                values_train.append(temp_train)
                values_test.append(temp_test)

    # create table
    table_train = create_table(values_train, col_labels, '', 'train', sort_by=['combinator'], max_=True)
    table_test = create_table(values_test, col_labels, '', 'test', sort_by=['combinator'], max_=True)

    # save table
    save_table(table_train, table_test, save_path, 'best')


def plot_table_attention(path_dict, save_path):
    res_json = ['results.json', 'results_hr.json']
    for i, act in enumerate(path_dict.keys()):
        row_labels, values_train, values_test = [], [], []
        if act not in COMBINED_ACT:
            continue
        for path in path_dict[act]:
            # print(act)
            for res in res_json:
                try:
                    with open(f'{path}/{res}', 'r') as f:
                        results = json.load(f)
                        # att = 1 if res == 'results_hr.json' else 0
                except Exception as e:
                    continue
                if results['combinator'] not in ATT_LIST:
                    continue
                if i == 0:
                    col_labels = fill_col_labels(results, att=1)
                temp_train, temp_test = fill_row_values(results, path, act, att=1)
                values_train.append(temp_train)
                values_test.append(temp_test)

        # create table
        table_train = create_table(values_train, col_labels, act, 'train',
                                   sort_by=['combinator'], att=1)
        table_test = create_table(values_test, col_labels, act, 'test', sort_by=['combinator'], att=1)

        # save table
        save_table(table_train, table_test, save_path + 'ATT_', act)


# ============== AUXILIARY METHODS ===================

def compute_distance(results, path):
    if results['combinator'] in MLP_LIST+ATT_LIST:
        mix = MIX(results['act_fn'],
                  combinator=results['combinator'],
                  neurons=neurons,
                  normalize=results['normalize'])
        o1 = compute_MLP_activations(results, '1', mix, path)
        o2 = compute_MLP_activations(results, '200', mix, path)
    elif results['combinator'] == 'Linear':
        o1 = compute_Linear_activations(results, '1')
        o2 = compute_Linear_activations(results, '200')
    elif results['combinator'] == 'Kwinners':
        return 0, neurons - results['k']
    else:
        return 0, 0
    # print(results['combinator'], o1.shape, o2.shape, torch.norm((o1 - o2), dim=-1).shape)
    distance = torch.mean((torch.norm((o1 - o2), dim=-1))).item()
    zeroed_n = torch.sum(
        torch.where(torch.norm(o2, dim=-1) / n_samples <= 0.001, torch.tensor(1), torch.tensor(0))).item()
    return distance, zeroed_n


def fill_col_labels(results, max_=False, att=False):
    col_labels = ['combinator']
    if max_:
        col_labels.append('act')
    col_labels.append('d(a₀,aₙ)')
    col_labels.append('zeroed neurons')
    col_labels.append('drop')
    col_labels.append('normaliz')
    if att == 0:
        col_labels.append('init')
    elif att == 1:
        col_labels.append('H.R.')
    elif att == 2:
        col_labels.append('init')
        col_labels.append('H.R.')
    col_labels.append('max acc')
    col_labels.append('min acc')
    for epoch in list(results['train_acc'].keys()):
        if int(epoch) % n_epochs == 0:
            col_labels.append(f'e:[{int(epoch) - n_epochs},{epoch})')
    return col_labels


def fill_row_values(results, path, act='', max_=False, att=0):
    values_train_temp = [results['combinator']]
    if max_:
        temp = act.split('_')
        values_train_temp.append('.'.join(temp[i][0] for i in range(len(temp))).upper())  # act
    norm_diff, zeroed_n = compute_distance(results, path) if not results.get('hr', False) else (0, 0)
    values_train_temp.append(norm_diff)  # 1/p * sum(norm(act_1 - act_f))
    values_train_temp.append(zeroed_n)
    alpha_dropout = f'{results.get("alpha_dropout","-")}'
    values_train_temp.append(alpha_dropout if alpha_dropout != 'None' else '-')
    values_train_temp.append(results['normalize'] if results['normalize'] != 'None' else '-')
    if att == 0:
        values_train_temp.append(results['init'] if results['init'] != 'None' else '-')
    elif att == 1:
        values_train_temp.append('✔' if results.get('hr', False) is True else '✗')
    elif att == 2:
        values_train_temp.append(results['init'] if results['init'] != 'None' else '-')
        values_train_temp.append('✔' if results.get('hr', False) is True else '✗')

    values_test_temp = values_train_temp.copy()

    values_train_temp.append(max(list(results['train_acc'].values())) / 100)
    values_train_temp.append(min(list(results['train_acc'].values())[40:]) / 100)
    values_test_temp.append(max(list(results['test_acc'].values())) / 100)
    values_test_temp.append(min(list(results['test_acc'].values())[40:]) / 100)
    count_train, count_test = 0, 0
    for epoch in list(results['train_acc'].keys()):
        count_train += results['train_acc'][epoch] / 100
        count_test += results['test_acc'][epoch] / 100
        if int(epoch) % n_epochs == 0:
            values_train_temp.append(count_train / n_epochs)
            values_test_temp.append(count_test / n_epochs)
            count_train, count_test = 0, 0
    return values_train_temp, values_test_temp


def highlight_max(data, mode='train'):
    df = data.copy()
    attr = {'best': 'background-color: {}'.format('#fc6e6e'),
            'first': 'background-color: {}'.format('#ffad33'),
            'second': 'background-color: {}'.format('#ffe717'),
            'third': 'background-color: {}'.format('#fef5a1')
            }

    cond_best = df >= df.max().max()

    if mode == 'test':
        cond = [df >= 0.951, df >= 0.948, df >= 0.944]
        for col in df:
            if col == 'max acc':
                cond = [df >= 0.957, df >= 0.956, df >= 0.955]
            elif col == 'min acc':
                cond = [df >= 0.930, df >= 0.920, df >= 0.910]
    else:
        cond = [df >= 0.999, df >= 0.995, df >= 0.990]
        for col in df:
            if col == 'max acc':
                cond = [df >= 0.999, df >= 0.996, df >= 0.993]
            elif col == 'min acc':
                cond = [df >= 0.991, df >= 0.985, df >= 0.980]

    return pd.DataFrame(np.where(cond_best, attr['best'],
                                 (np.where(cond[0], attr['first'],
                                           (np.where(cond[1], attr['second'],
                                                     (np.where(cond[2], attr['third'], '')
                                                      )))))),
                        index=data.index, columns=data.columns)


def apply_color(data):
    attr = {'normal': 'background-color: {}'.format('#93c9e6'),
            'uniform': 'background-color: {}'.format('#93e6c5'),
            'random': 'background-color: {}'.format('#d7e8a5'),
            'None': 'background-color: {}'.format('#92ada2'),
            'Sigmoid': 'background-color: {}'.format('#b3e6ff'),
            'Softmax': 'background-color: {}'.format('#ffc2b3'),
            'Tanh': 'background-color: {}'.format('#887fee'),
            'A.I.R.S': 'background-color: {}'.format('#f9e79f'),
            'A.I.R.S.T': 'background-color: {}'.format('#f5cba7'),
            'A.I.R.T': 'background-color: {}'.format('#abebc6'),
            'A.I.S.T': 'background-color: {}'.format('#aed6f1'),
            'A.R.S.T': 'background-color: {}'.format('#d2b4de'),
            'I.R.S.T': 'background-color: {}'.format('#abb2b9'),
            'Linear': 'background-color: {}'.format('#22c186'),
            'MLP1': 'background-color: {}'.format('#81f4f4'),
            'MLP2': 'background-color: {}'.format('#f4e881'),
            'MLP3': 'background-color: {}'.format('#8186f4'),
            'MLP4': 'background-color: {}'.format('#f481bd'),
            'MLP5': 'background-color: {}'.format('#c39ef5'),
            'MLPr': 'background-color: {}'.format('#f4b181'),
            'MLP_ATT': 'background-color: {}'.format('#e16b5b'),
            'MLP_ATT_neg': 'background-color: {}'.format('#a57d7d ')
            }

    if data.ndim == 1:
        return [attr[v] if v in attr.keys() else '' for v in data]


def create_table(values, col_labels, act, s, max_=False, sort_by=None, att=0):
    """
    - create a dataframe with given infos (values and column labels)
    - create a table style:
        - color the 'norm' column values
        - highlights all the cells that match the 3 best accuracies reached by all the experiments
          (best: orange, 2nd best: yellow, 3rd best: lightyellow)
        - adjust the table style to increase readability
    """
    sort_by = ['combinator', 'normaliz', 'drop'] if att != 0 else ['combinator', 'normaliz', 'init', 'drop']
    table = pd.DataFrame(values, columns=col_labels).sort_values(by=sort_by)
    cm_green = sns.light_palette("green", as_cmap=True)
    subset_to_color = ['act', 'combinator', 'normaliz', 'init', 'drop']
    if max_ is False:
        subset_to_color = subset_to_color[1:]
    if att != 0:
        subset_to_color = ['combinator']
    styled_table = table.style.apply(apply_color, subset=subset_to_color, axis=0). \
        apply(highlight_max, axis=None, subset=col_labels[-10:], mode=s). \
        apply(highlight_max, axis=None, subset=['min acc'], mode=s). \
        apply(highlight_max, axis=None, subset=['max acc'], mode=s). \
        hide_index(). \
        background_gradient(subset='d(a₀,aₙ)', cmap=cm_green). \
        background_gradient(subset=['zeroed neurons'], cmap=cm_green). \
        set_properties(**{'width': '250px'}, **{'text-align': 'center'}). \
        set_caption(act + ' [{}]'.format(s.upper())). \
        set_table_styles([{'selector': 'caption', 'props': [('color', 'black'), ('font-size', '25px')]},
                          {'selector': 'tr:nth-of-type(odd)', 'props': [('background', '#eee')]},
                          {'selector': 'tr:nth-of-type(even)', 'props': [('background', 'white')]},
                          {'selector': 'th', 'props': [('background', '#606060'), ('color', 'white')]}])
    return styled_table


def save_table(table_train, table_test, save_path, act):
    # save the table as .png file
    html_train = '<meta charset="UTF-8">' + table_train.render()
    html_test = '<meta charset="UTF-8">' + table_test.render()
    html = html_train + html_test
    imgkit.from_string(html, save_path + "{}.png".format(act))


def create_path_dict(save_path):
    """
    will create a dictionary with MIX of activation functions as keys(),
    and as values() a list of the paths where their result.json file are.

    eg: path_dict = {'antirelu_identity_relu_sigmoid': ['./res/init_normal/norm_None/0/',
                                                './res/init_normal/norm_None/0.1/',
                                                ...],
                    'antirelu_identity_relu_sigmoid_tanh': [...]
                    }
    """
    act_fn = [sorted(['relu', 'antirelu', 'identity', 'tanh', 'sigmoid']),
              sorted(['relu', 'antirelu', 'identity', 'sigmoid']),
              sorted(['relu', 'antirelu', 'identity', 'tanh']),
              sorted(['relu', 'antirelu', 'sigmoid', 'tanh']),
              sorted(['relu', 'identity', 'sigmoid', 'tanh']),
              sorted(['antirelu', 'identity', 'sigmoid', 'tanh']),
              ['relu'],
              ['sigmoid'],
              ['tanh'],
              ['antirelu'],
              ['None']]
    # ['identity']]

    act_fn = ['_'.join(act) for act in act_fn]
    path_dict = defaultdict(list)
    for (filepath, dirname, filename) in os.walk(save_path):
        if 'results.json' in filename:
            for act in act_fn:
                temp = filepath.split('/')
                if act == temp[-1] or act == temp[-2]:
                    path_dict[act].append(filepath)
    return path_dict


def fix_json():
    norm = {"None": 'none',
            "Softmax": 'soft',
            "Sigmoid": 'sigm'
            }
    init = {"None": 'none',
            "random": 'rand',
            "uniform": 'unif',
            "normal": 'norm'}
    for (filepath, dirname, filename) in os.walk('../experiments/MNIST/'):
        if 'results.json' in filename:
            # temp = filepath.split('/')[1]
            # print(filepath)
            with open(filepath + '/results.json', 'r') as f:
                results = json.load(f)
            # results['dataset'] = 'MNIST'
            # results['run_name'] = "{}_{}_{}".format(results['dataset'], results['combinator'], results['random_seed'])

            if results['combinator'] in ['MLP_ATT_neg', 'MLP_ATT']:
                results['run_name'] += '_'+str(results['alpha_dropout'])
            with open(filepath + '/results.json', 'w') as f:
                json.dump(results, f, indent=4)


# ========================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-table', action='store_true')
    parser.add_argument('-table_att', action='store_true')
    parser.add_argument('-table_max', action='store_true')
    parser.add_argument('-accuracy', action='store_true')
    parser.add_argument('-activations', action='store_true')
    parser.add_argument('--run_name', type=str)
    args = parser.parse_args()
    # fix_json()

    source_path = '../experiments/MNIST/'
    imgs_path = '../experiments/MNIST/imgs/'

    path_dict = create_path_dict(source_path)
    # print(path_dict.keys())

    if args.table:
        print('plotting table...')
        plot_table(path_dict, imgs_path)

    if args.table_att:
        print('plotting table...')
        plot_table_attention(path_dict, imgs_path)

    if args.table_max:
        print('plotting table max...')
        plot_table_max(path_dict, imgs_path, 0.956)

    if args.accuracy:
        print('plotting accuracy...')
        plot_accuracy(path_dict, imgs_path)

    if args.activations:
        print('plotting activations...')
        plot_activations(path_dict)
