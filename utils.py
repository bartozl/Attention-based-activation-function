import json
from pathlib import Path
import sys
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from matplotlib import pyplot as plt
from collections import defaultdict
import os
import seaborn as sns
import imgkit
import pandas as pd
import numpy as np
import random
from mixed_activations import MIX
from modules import Network

MLP_LIST = ['MLP1', 'MLP1_neg', 'MLP2', 'MLP3', 'MLP4', 'MLP5', 'MLPr']
ATT_LIST = ['MLP_ATT', 'MLP_ATT_neg', 'MLP_ATT_b']
MLP_neg = ['MLP1_neg', 'MLP_ATT_neg']


# ============================================= #
# ======= EXECUTION AUXILIARY FUNCTIONS ======= #
# ============================================= #


def create_save_dir(run_config, init, normalize, act_list, lambda_l1, drop, hr_test):
    acts = '_'.join(act_list)
    if run_config['combinator'] == 'Linear':
        save_dir = f'../experiments/{run_config["dataset"]}/{run_config["combinator"]}' \
                   f'/init_{init}/norm_{normalize}/{acts}/{lambda_l1}/'
    elif run_config['combinator'] in MLP_LIST:
        save_dir = f'../experiments/{run_config["dataset"]}/MLP/{run_config["combinator"]}' \
                   f'/init_None/norm_{normalize}/{acts}/{lambda_l1}/'
    elif run_config['combinator'] == 'Kwinners':
        save_dir = f'../experiments/{run_config["dataset"]}/' \
                   f'{run_config["combinator"]}/{run_config["k"]}/'
    elif run_config['combinator'] in ATT_LIST:
        save_dir = f'../experiments/{run_config["dataset"]}/ATT/{run_config["combinator"]}' \
                   f'/init_None/norm_{normalize}/drop_{drop}/{acts}/{lambda_l1}/'
    else:
        print('ERROR: unknown combinator')
        sys.exit(0)
    if Path(f'{save_dir}results.json').exists():
        if hr_test is not None:
            return save_dir
    #    else:
    #        return False
    Path(f'{save_dir}weights/').mkdir(parents=True, exist_ok=True)
    Path(f'{save_dir}plot/').mkdir(parents=True, exist_ok=True)
    shutil.copy2('run_config.json', save_dir)
    return save_dir


def load_dataset(dataset_name, subset, batch_size, colab):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        ])
        if colab:
            data_path = '~/../content/'
        else:
            data_path = '../datasets/'
        train_dataset = datasets.MNIST(data_path, download=True, train=True, transform=transform)
        test_dataset = datasets.MNIST(data_path, download=True, train=False, transform=transform)
        len_train, len_test = int(len(train_dataset) * subset), int(len(test_dataset) * subset)
        subset_indices_train, subset_indices_test = range(len_train), range(len_test)
        data_train = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(subset_indices_train))
        data_test = DataLoader(test_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(subset_indices_test))
    else:
        print('unknown dataset')
        sys.exit(0)
    return data_train, data_test, len_train, len_test


def load_run_config(run_config):
    with open(run_config, 'r') as f:
        run_config = json.load(f)
    run_config['run_name'] = f"{run_config['combinator']}"
    if run_config['combinator'] == 'Kwinners':
        run_config['run_name'] += f"{run_config['k']}"
    return run_config

def reset_seed(seed):
    np.random.seed(seed)  # allows reproducibility
    torch.manual_seed(seed)  # allows reproducibility
    random.seed(seed)


def generate_configs(run_config, hr_test, colab):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    savedirs, configs = [], []
    data_train, data_test, len_train, len_test = load_dataset(run_config['dataset'],
                                                              run_config['subset'],
                                                              run_config['batch_size'],
                                                              colab)
    for norm_ in run_config['normalize']:
        for act_ in run_config['act_fn']:
            for lamb_ in run_config['lambda_l1']:
                for init_ in run_config['init']:
                    for drop_ in run_config['alpha_dropout']:
                        reset_seed(run_config['random_seed'])
                        save_dir = create_save_dir(run_config, init_, norm_, act_, lamb_, drop_, hr_test)
                        if (save_dir is False) or (save_dir in savedirs):
                            continue
                        savedirs.append(save_dir)
                        # if run_config['dataset'] == 'MNIST':
                        network = Network(act_, run_config['combinator'], norm_, init_, drop_, hr_test).to(device)
                        config = {'save_dir': save_dir,
                                  'data_test': data_test,
                                  'data_train': data_train,
                                  'len_train': len_train,
                                  'len_test': len_test,
                                  'act_fn': act_,
                                  'normalize': norm_,
                                  'init': init_,
                                  'lambda_l1': lamb_,
                                  'neurons': 128,
                                  'network': network,
                                  'optimizer': torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=1e-4),
                                  'device': device,
                                  'save_every': 1,
                                  'combinator': run_config['combinator'],
                                  'run_name': run_config['run_name'],
                                  'random_seed': run_config['random_seed'],
                                  'batch_size': run_config['batch_size'],
                                  'dataset': run_config['dataset'],
                                  'epochs': run_config['epochs'],
                                  'k': run_config['k'] if run_config['combinator'] == 'Kwinners' else None,
                                  'alpha_dropout': drop_ if run_config['combinator'] in ATT_LIST else None
                                  }
                        configs.append(config)
    return configs


def save_state(config, dest_path, acc, train_=True):
    if train_:
        len_dataset = config['len_train']
        state_dict = config['network'].state_dict()
        torch.save(state_dict, dest_path)
    else:
        len_dataset = config['len_test']
    new_acc = 100 * acc / len_dataset
    return new_acc


def save_results(config, train_acc, test_acc, epoch, loss=None, hr_test=None):

    first_save = False
    try:
        with open(f'{config["save_dir"]}results.json', 'r') as f:
            results = json.load(f)
    except Exception as e:
        first_save = True

    if first_save:
        results = {'act_fn': config['act_fn'],
                   'epochs': config['epochs'],
                   'act_per_neuron': config['neurons'] if config['neurons'] != 0 else None,
                   'train_acc': {1: train_acc},
                   'test_acc': {1: None},
                   'loss': {1: loss},
                   'lambda_l1': config['lambda_l1'],
                   'normalize': config['normalize'],
                   'combinator': config['combinator'],
                   'run_name': config['run_name'],
                   'dataset': config['dataset'],
                   'random_seed': config['random_seed'],
                   'batch_size': config['batch_size'],
                   'init': config['init'],
                   'k': config['k'],
                   'alpha_dropout': config['alpha_dropout']
                   }
    else:
        if hr_test is None:
            if train_acc is None:
                results['test_acc'][epoch] = test_acc
            if test_acc is None:
                results['train_acc'][epoch] = train_acc
                results['loss'][epoch] = loss
        else:
            if f'test_acc_hr_{hr_test}' not in results:
                results[f'test_acc_hr_{hr_test}'] = {1: test_acc}
            else:
                results[f'test_acc_hr_{hr_test}'][epoch] = test_acc

    print('...Saving results...')
    with open(config['save_dir'] + 'results.json', 'w') as f:
        json.dump(results, f, indent=4)
    return results


# ======================================== #
# ======= PLOT AUXILIARY FUNCTIONS ======= #
# ======================================== #

device = 'cuda' if torch.cuda.is_available() else 'cpu'
neurons = 128
n_samples = 500
input_ = torch.Tensor(np.linspace(-2, 2, n_samples).astype(np.float32)).repeat(neurons, 1).to(device)


def grid_activations(dest_path, out, fixed_out, name, act, alpha, bias, nx=5, ny=4, sizex=32, sizey=18):
    tick_fontsize = int(100/nx)
    legend_fontsize = int(120/nx)
    lw = int(20/nx)
    fig, ax = plt.subplots(nx, ny)
    fig.set_size_inches(sizex, sizey)
    fig.tight_layout(pad=4)
    idx = 0
    mean_max = torch.mean(torch.max(alpha, dim=-1)[0]).item()
    print(f'{dest_path}/{name}  - {mean_max}')
    for i in range(nx):
        for j in range(ny):
            plt.setp(ax[i][j].get_xticklabels(), fontsize=tick_fontsize)
            plt.setp(ax[i][j].get_yticklabels(), fontsize=tick_fontsize)
            ax[i][j].set_xticks([-2, -1, 0, 1, 2])
            ax[i][j].set_yticks([-2, -1, 0, 1, 2])
            ax[i][j].set_ylim([-2, 2])
            ax[i][j].axhline(ls='--', color='lightgray')
            ax[i][j].axvline(ls='--', color='lightgray')
            if alpha is None:
                alpha_title = ""
            else:
                alpha_title = "".join(["", "+"][float(alpha[idx][i]) > 0] + str(np.round(float(alpha[idx][i]), 2))+a[0].upper() for i, a in enumerate(act.split("_")))
            b = f'{["","+"][bias[idx]>0]}{bias[idx]:.2f}' if bias is not None else ""
            ax[i][j].set_title(f'[{idx}] {alpha_title}{b}', fontsize=legend_fontsize)
            ax[i][j].plot(input_[idx].detach().numpy(), out[idx].detach().numpy(), color='green', lw=lw, alpha=0.8)
            ax[i][j].plot(input_[idx].detach().numpy(), fixed_out[idx].detach().numpy(), color='red', ls='--', lw=lw, alpha=0.8)
            idx += 1
    plt.savefig(f'{dest_path}/{name}.png', bbox_inches='tight', dpi=200)
    plt.close()


def grid_accuracy(results, label, ax, i, first=False, color='green'):
    title_fontsize = 45
    label_fontsize = 35
    tick_fontsize = 30
    legend_fontsize = 25
    alpha = 1
    lw = 3
    if first:
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

    ax[0][i].plot(list(results['train_acc'].keys()), list(results['train_acc'].values()),
                  color=color, label=label, alpha=1, linewidth=lw)
    ax[1][i].plot(list(results['test_acc'].keys()), list(results['test_acc'].values()),
                  label=label, color=color, alpha=alpha, linewidth=lw)

    ax[0][i].legend(fontsize=legend_fontsize, loc="best")
    ax[1][i].legend(fontsize=legend_fontsize, loc="best")

    for j, label_ax in enumerate(ax[0][i].xaxis.get_ticklabels()):
        if (j + 1) % 20 != 0:
            label_ax.set_visible(False)
    for j, label_ax in enumerate(ax[1][i].xaxis.get_ticklabels()):
        if (j + 1) % 20 != 0:
            label_ax.set_visible(False)

# TODO delete results['epoch'] key
def compute_activations(results, epoch, path):
    state_dict = torch.load(f'{path}/weights/{epoch}.pth')  # load the model of the whole original network
    state_dict_filt = {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items()}  # adjust the name of the parameters
    # print(state_dict)
    del state_dict_filt['weight']  # delete the parameter of the original network linear layers in order to keep only\
    del state_dict_filt['bias']  # the MIX layer parameters
    mix = MIX(results['act_fn'], results['combinator'], 128, results['normalize'], results['init'])
    mix.eval()  # evaluation mode
    mix.load_state_dict(state_dict_filt)  # load the MIX parameters
    # print(results["combinator"], 'output.shape', output.shape)

    output, alpha, bias, params = mix(input_.T)

    if results['combinator'] in ATT_LIST:
        alpha = alpha.permute(1, 0, 2)
        alpha = torch.sum(alpha, dim=1) / alpha.shape[1]

    return output.T, alpha, bias, params

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


def compute_distance(results, path):
    if results['combinator'] in MLP_LIST + ATT_LIST + MLP_neg + ['Linear']:
        o1 = compute_activations(results, '1', path)
        o2 = compute_activations(results, '200', path)
    elif results['combinator'] == 'Kwinners':
        return 0, neurons - results['k']
    else:
        return 0, 0
    # print(results['combinator'], o1.shape, o2.shape, torch.norm((o1 - o2), dim=-1).shape)
    distance = torch.mean((torch.norm((o1 - o2), dim=-1))).item()
    # zeroed_n = torch.sum(
    #    torch.where(torch.norm(o2, dim=-1) / n_samples <= 0.001, torch.tensor(1), torch.tensor(0))).item()
    # print((torch.sum(torch.abs(o2), dim=-1) / (n_samples * 128)))
    zeroed_n = torch.sum(
        torch.where((torch.sum(torch.abs(o2), dim=-1) / (n_samples * 128)) <= 0.0001,
                    torch.tensor(1), torch.tensor(0))).item()
    return distance, zeroed_n


def fill_col_labels(results, max_=False, att=False, n_epochs=20):
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


def fill_row_values(results, path, act='', max_=False, att=0, n_epochs=20):
    values_train_temp = [results['combinator']]
    if max_:
        temp = act.split('_')
        values_train_temp.append('.'.join(temp[i][0] for i in range(len(temp))).upper())  # act
    norm_diff, zeroed_n = compute_distance(results, path) if not results.get('hr', False) else (0, 0)
    values_train_temp.append(norm_diff)  # 1/p * sum(norm(act_1 - act_f))
    values_train_temp.append(zeroed_n)
    alpha_dropout = f'{results.get("alpha_dropout", "-")}'
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


def create_table(values, col_labels, act, s, max_=False, att=0):
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
                results['run_name'] += '_' + str(results['alpha_dropout'])
            with open(filepath + '/results.json', 'w') as f:
                json.dump(results, f, indent=4)
