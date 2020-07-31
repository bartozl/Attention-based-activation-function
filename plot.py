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
# TODO set n_epochs = 20
n_epochs = 20


def plot_activations(path_dict):
    fixed_output = None  # store the first epoch activation i.o.t. compare it with acts of the others epochs
    alpha, bias = None, None
    for act in path_dict:
        for i, path in enumerate(path_dict[act]):
            with open(path + '/results.json', 'r') as f:
                results = json.load(f)
            dest_path = f'{path}/plot/'  # where the imgs will be saved
            # print(results['combinator'], act)
            if (plot_it is not None) and (results['combinator'] not in plot_it):
                continue
            if len(os.listdir(dest_path)) == 12:  # imgs for epochs in [1, 20, 40, ... 200] + 1 img with all neurons'act
                continue
            print(path)
            for epoch in results['train_acc'].keys():
                if int(epoch) % n_epochs != 0 and int(epoch) != 1:  # plot every 20 epochs (first epoch included)
                    continue
                if results['combinator'] in MLP_LIST + MLP_neg + ATT_LIST + ['Linear'] + ['None']:
                    output, alpha, bias, _ = utils.compute_activations(results, epoch, path)
                else:
                    print(f"no plot method available for {results['combinator']} combinator...")
                    continue
                if int(epoch) == 1:  # store the first epoch activation
                    fixed_output = output
                utils.grid_activations(dest_path, output, fixed_output, epoch, act, alpha, bias, results)
                if int(epoch) == 200:
                    utils.grid_activations(dest_path, output, fixed_output, epoch+'_tot', act, alpha, bias, results, nx=11, ny=11)


def plot_accuracy(path_dict, dest_path):
    path_dict_ = path_dict.copy()
    pairs_to_compare = [['MLP1', 'MLP_ATT_neg']]  # it must be a list of lists!
    for relu_path in path_dict_['relu']:
        with open(relu_path + '/results.json', 'r') as f:
            results_relu = json.load(f)
        if results_relu['batch_size'] == 256:
            break
    del path_dict_['relu']  # , path_dict_['sigmoid'], path_dict_['antirelu'], path_dict_['tanh']
    for pair in pairs_to_compare:
        for act in path_dict_.keys():
            Path(f'{dest_path}/{act}/').mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(2, 2)
            fig.tight_layout()
            fig.set_size_inches(32, 18)
            utils.grid_accuracy(results_relu, 'relu', ax, 0, first=True, color='blue')
            utils.grid_accuracy(results_relu, 'relu', ax, 1, first=True, color='blue')
            i = 0
            for path in sorted(path_dict_[act]):
                with open(path + '/results.json', 'r') as f:
                    results = json.load(f)
                if results['run_name'] not in pair:
                    continue
                color = 'green' if results['run_name'] == pair[0] else 'red'
                utils.grid_accuracy(results, results['run_name'], ax, i, color=color)
                i += 1
            print(dest_path + '/{}/{}_vs_{}.png'.format(act, pair[0], pair[1]))
            plt.savefig(dest_path + '/{}/{}_vs_{}.png'.format(act, pair[0], pair[1]))
            plt.close()


def plot_table(path_dict, save_path):
    pd.set_option('display.precision', 4)
    pd.set_option('display.width', 40)
    pd.set_option('display.float_format', '{:,.3f}'.format)

    for i, act in enumerate(path_dict.keys()):
        row_labels, values_train, values_test = [], [], []
        for path in path_dict[act]:
            with open(path + '/results.json', 'r') as f:
                results = json.load(f)
                if (plot_it is not None) and (results['combinator'] not in plot_it):
                    continue
                if i == 0:
                    col_labels = utils.fill_col_labels(results)
                temp_train, temp_test = utils.fill_row_values(results, path, act)
                values_train.append(temp_train)
                values_test.append(temp_test)
                if 'test_acc_hr_0.0' in results:
                    temp_train, temp_test = utils.fill_row_values(results, path, act,hr=0.0)
                    values_train.append(temp_train)
                    values_test.append(temp_test)

        # create table
        table_train = utils.create_table(values_train, col_labels, act, 'train')
        table_test = utils.create_table(values_test, col_labels, act, 'test')
        # save table
        utils.save_table(table_train, table_test, save_path, act)


def plot_table_max(path_dict, save_path, limit):
    row_labels, values_train, values_test = [], [], []
    for i, act in enumerate(path_dict.keys()):
        for path in path_dict[act]:
            with open(f'{path}/results.json', 'r') as f:
                results = json.load(f)
            if (plot_it is not None) and (results['combinator'] not in plot_it):
                continue
            if i == 0:
                col_labels = utils.fill_col_labels(results, max_=True, att=2)
            temp_train, temp_test = utils.fill_row_values(results, path, act, max_=True, att=2)
            # print(temp_test[9])
            if True not in np.where(temp_test[10] >= limit, True, False):
                continue
            values_train.append(temp_train)
            values_test.append(temp_test)

    # create table
    table_train = utils.create_table(values_train, col_labels, '', 'train', max_=True)
    table_test = utils.create_table(values_test, col_labels, '', 'test', max_=True)

    # save table
    utils.save_table(table_train, table_test, save_path, 'best')


def plot_table_attention(path_dict, save_path):
    for i, act in enumerate(path_dict.keys()):
        row_labels, values_train, values_test = [], [], []
        if act not in COMBINED_ACT:
            continue
        for path in path_dict[act]:
            # print(act)
            with open(f'{path}/results.json', 'r') as f:
                results = json.load(f)
            if results['combinator'] not in ATT_LIST or results['combinator'] not in plot_it:
                continue
            if i == 0:
                col_labels = utils.fill_col_labels(results, att=1)
            temp_train, temp_test = utils.fill_row_values(results, path, act, att=1)
            values_train.append(temp_train)
            values_test.append(temp_test)
            if 'test_acc_hr_0.0' in results:
                temp_train, temp_test = utils.fill_row_values(results, path, act, att=1, hr=0.0)
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
    parser.add_argument('--dataset', type=str, required=True, choices=['MNIST', 'CIFAR10'])
    args = parser.parse_args()
    # utils.fix_json()
    np.random.seed(1)  # allows reproducibility

    dataset = args.dataset
    source_path = f'../experiments/{dataset}/'
    imgs_path = f'{source_path}imgs/'

    print(source_path, imgs_path)

    path_dict = utils.create_path_dict(source_path)
    # print(path_dict.keys())
    plot_it = ['None', 'MLP_ATT', 'MLP_ATT_neg', 'MLP1', 'MLP2', 'Linear', 'MLP_ATT_b']

    if args.table:
        print('plotting table...')
        plot_table(path_dict, imgs_path)

    if args.table_att:
        print('plotting table...')
        plot_table_attention(path_dict, imgs_path)

    if args.table_max:
        max_value = 0.54 if dataset == 'CIFAR10' else 0.982
        print('plotting table max...')
        plot_table_max(path_dict, imgs_path, max_value)

    if args.accuracy:
        print('plotting accuracy...')
        plot_accuracy(path_dict, imgs_path)

    if args.activations:
        print('plotting activations...')
        plot_activations(path_dict)

# ['MLP1', 'MLP2', 'Linear', 'MLP_ATT', 'MLP_ATT_neg'] <-- already plotted
