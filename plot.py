import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from torch import nn
from mixed_activations import Antirelu, Identity
import numpy as np
from pathlib import Path
import utils


MLP_LIST = ['MLP1', 'MLP1_neg', 'MLP2', 'MLP3', 'MLP4', 'MLP5', 'MLPr']
ATT_LIST = ['MLP_ATT', 'MLP_ATT_neg', 'MLP_ATT_b']
MLP_neg = ['MLP1_neg', 'MLP_ATT_neg']
COMBINED_ACT = ['antirelu_identity_sigmoid_tanh', 'antirelu_identity_relu_sigmoid', 'identity_relu_sigmoid_tanh']
act_module = {'relu': nn.ReLU(),  # dictionary containing useful functions
              'sigmoid': nn.Sigmoid(),
              'tanh': nn.Tanh(),
              'antirelu': Antirelu(),
              'identity': Identity(),
              'softmax': nn.Softmax()}
n_epochs = 20


def plot_activations(path_dict):
    fixed_output = None  # store the first epoch activation i.o.t. compare it with acts of the others epochs
    for act in path_dict:
        for path in path_dict[act]:
            with open(path + '/results.json', 'r') as f:
                results = json.load(f)
            dest_path = f'{path}/plot/'  # where the imgs will be saved
            if len(os.listdir(dest_path)) == 12:  # imgs for epochs in [1, 20, 40, ... 200] + 1 img with all neurons
                continue
            print(path)
            for epoch in results['train_acc'].keys():
                if int(epoch) % n_epochs != 0 and int(epoch) != 1:  # plot every 20 epochs (first epoch included)
                    continue
                if results['combinator'] in MLP_LIST+ATT_LIST+MLP_neg+['Linear']:
                    output = utils.compute_activations(results, epoch, path)
                else:
                    print(f"no plot method available for {results['combinator']} combinator...")
                    continue
                if int(epoch) == 1:  # store the first epoch activation
                    fixed_output = output
                utils.save_grid_fig(dest_path, output, fixed_output, epoch)
                if int(epoch) == 200:
                    utils.save_grid_fig(dest_path, output, fixed_output, epoch, nx=11, ny=11)


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
    pd.set_option('display.precision', 4)
    pd.set_option('display.width', 40)

    for i, act in enumerate(path_dict.keys()):
        row_labels, values_train, values_test = [], [], []
        for path in path_dict[act]:
            with open(path + '/results.json', 'r') as f:
                results = json.load(f)
                if i == 0:
                    col_labels = utils.fill_col_labels(results)
                temp_train, temp_test = utils.fill_row_values(results, path, act)
                values_train.append(temp_train)
                values_test.append(temp_test)

        # create table
        table_train = utils.create_table(values_train, col_labels, act, 'train')
        table_test = utils.create_table(values_test, col_labels, act, 'test')
        # save table
        utils.save_table(table_train, table_test, save_path, act)


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
                if i == 0:
                    col_labels = utils.fill_col_labels(results, max_=True, att=2)
                temp_train, temp_test = utils.fill_row_values(results, path, act, max_=True, att=2)
                if True not in np.where(temp_test[8] >= limit, True, False):
                    continue
                values_train.append(temp_train)
                values_test.append(temp_test)

    # create table
    table_train = utils.create_table(values_train, col_labels, '', 'train', max_=True)
    table_test = utils.create_table(values_test, col_labels, '', 'test', max_=True)

    # save table
    utils.save_table(table_train, table_test, save_path, 'best')


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
                    col_labels = utils.fill_col_labels(results, att=1)
                temp_train, temp_test = utils.fill_row_values(results, path, act, att=1)
                values_train.append(temp_train)
                values_test.append(temp_test)

        # create table
        table_train = utils.create_table(values_train, col_labels, act, 'train', att=1)
        table_test = utils.create_table(values_test, col_labels, act, 'test', att=1)
        # save table
        utils.save_table(table_train, table_test, save_path + 'ATT_', act)


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
    # utils.fix_json()

    source_path = '../experiments/MNIST/'
    imgs_path = '../experiments/MNIST/imgs/'

    path_dict = utils.create_path_dict(source_path)
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
