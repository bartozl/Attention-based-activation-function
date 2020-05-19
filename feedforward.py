import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from mixed_activations import MIX
from Kwinners import Kwinners
import json
import os
from collections import OrderedDict
import shutil
import argparse
import sys
from time import time
from datetime import timedelta
from pathlib import Path

MLP_list = ['MLP1', 'MLP1_neg', 'MLP2', 'MLP3', 'MLP4', 'MLP5', 'MLPr']
ATT_LIST = ['MLP_ATT', 'MLP_ATT_b', 'MLP_ATT_neg']


def load_run_config(run_config):
    with open(run_config, 'r') as f:
        run_config = json.load(f)
    run_config['run_name'] = f"{run_config['combinator']}"
    if run_config['combinator'] == 'Kwinners':
        run_config['run_name'] += f"{run_config['k']}"
    return run_config


def create_save_dir(run_config, init, normalize, act_list, lambda_l1, drop):
    acts = '_'.join(act_list)
    if run_config['combinator'] == 'Linear':
        save_dir = f'../experiments/{run_config["dataset"]}/{run_config["combinator"]}' \
                   f'/init_{init}/norm_{normalize}/{acts}/{lambda_l1}/'
    elif run_config['combinator'] in MLP_list:
        save_dir = f'../experiments/{run_config["dataset"]}/{run_config["combinator"]}' \
                   f'/init_None/norm_{normalize}/{acts}/{lambda_l1}/'
    elif run_config['combinator'] == 'Kwinners':
        save_dir = f'../experiments/{run_config["dataset"]}/' \
                   f'{run_config["combinator"]}/{run_config["k"]}/'
    elif run_config['combinator'] in ATT_LIST:
        save_dir = f'../experiments/{run_config["dataset"]}/{run_config["combinator"]}' \
                   f'/init_None/norm_{normalize}/drop_{drop}/{acts}/{lambda_l1}/'
    else:
        print('ERROR: unknown combinator')
        sys.exit(0)
    if Path(f'{save_dir}results.json').exists():
        if test_only:
            return save_dir
        else:
            return False
    Path(f'{save_dir}weights/').mkdir(parents=True, exist_ok=True)
    Path(f'{save_dir}plot/').mkdir(parents=True, exist_ok=True)
    shutil.copy2('run_config.json', save_dir)
    return save_dir


def load_dataset(dataset_name, subset, batch_size):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        ])
        train_dataset = datasets.MNIST('../datasets/', download=True, train=True, transform=transform)
        test_dataset = datasets.MNIST('../datasets/', download=True, train=False, transform=transform)
        len_train, len_test = int(len(train_dataset) * subset), int(len(test_dataset) * subset)
        subset_indices_train, subset_indices_test = range(len_train), range(len_test)
        data_train = DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(subset_indices_train))
        data_test = DataLoader(test_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(subset_indices_test))
    else:
        print('unknown dataset')
        sys.exit(0)
    return data_train, data_test, len_train, len_test


def generate_configs(run_config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    savedirs, configs = [], []
    data_train, data_test, len_train, len_test = load_dataset(run_config['dataset'],
                                                              run_config['subset'],
                                                              run_config['batch_size'])
    for norm_ in run_config['normalize']:
        for act_ in run_config['act_fn']:
            for lamb_ in run_config['lambda_l1']:
                for init_ in run_config['init']:
                    for drop_ in run_config['alpha_dropout']:
                        save_dir = create_save_dir(run_config, init_, norm_, act_, lamb_, drop_)
                        if (save_dir is False) or (save_dir in savedirs):
                            continue
                        savedirs.append(save_dir)
                        # if run_config['dataset'] == 'MNIST':
                        network = torch.nn.Sequential(
                            torch.nn.Linear(784, 128),
                            MIX(act_, run_config['combinator'],
                                neurons=128, normalize=norm_,
                                alpha_dropout=drop_).to(device),
                            # Kwinners(neurons=128, k=run_config['k']),
                            # torch.nn.ReLU(),
                            torch.nn.Linear(128, 10),
                            torch.nn.LogSoftmax(dim=1)).to(device)

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


def save_state(config, epoch, acc, train_=True):
    len_dataset = config['len_test']
    alpha_params = {}
    if config['combinator'] == 'Linear':
        for name, param in config['network'].named_parameters():
            if name == '1.alpha':
                alpha = param.data
        if len(alpha.shape) == 1:
            alpha = alpha.unsqueeze(0)
        for i, act in enumerate(config['act_fn']):
            alpha_params[act] = list(map(lambda l: float(l), alpha.T[i]))
            print(alpha_params[act][0])
    if train_:
        len_dataset = config['len_train']
        torch.save(config['network'].state_dict(), f'{config["save_dir"]}/weights/{epoch}.pth')
    new_acc = 100 * acc / len_dataset
    return new_acc, alpha_params


def save_results(config, train_acc_per_epoch, test_acc_per_epoch, alpha_per_epoch):
    if save_test:
        print('...Saving results_hr...')
        with open(config['save_dir'] + 'results.json', 'r') as f:
            results = json.load(f)
        results['test_acc'] = test_acc_per_epoch
        results['hr'] = True
        with open(config['save_dir'] + 'results_hr.json', 'w') as f:
            json.dump(results, f, indent=4)
        return results
    if test_only:
        return
    else:
        results = {'act_fn': config['act_fn'],
                   'epochs': config['epochs'],
                   'act_per_neuron': config['neurons'] if config['neurons'] != 0 else None,
                   'train_acc': train_acc_per_epoch,
                   'test_acc': test_acc_per_epoch,
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

        if config['combinator'] == 'Linear':
            results['alpha_per_epoch'] = alpha_per_epoch
        print('...Saving results...')
        with open(config['save_dir'] + 'results.json', 'w') as f:
            json.dump(results, f, indent=4)
    return results


def train(config):
    alpha_per_epoch = OrderedDict()  # used only for Linear combinator
    train_acc_per_epoch = OrderedDict()
    print('number of parameters', sum(p.numel() for p in config['network'].parameters() if p.requires_grad))
    for epoch in range(1, config['epochs'] + 1):
        time1 = time()
        config['network'].train()
        acc = 0
        for idx, (X_batch, y_batch) in enumerate(config['data_train']):
            config['optimizer'].zero_grad()
            X_batch = X_batch.to(config['device']).view(X_batch.shape[0], -1)
            y_batch = y_batch.to(config['device'])
            y_pred = config['network'](X_batch).to(config['device'])
            loss = F.nll_loss(y_pred, y_batch)
            loss.backward()
            config['optimizer'].step()
            predicted = torch.argmax(y_pred.data, 1)
            acc += (predicted == y_batch).sum().item()
        if epoch % config['save_every'] == 0:
            train_acc, alpha_params = save_state(config, epoch, acc)  # alpha_params only used for Linear combinator
            train_acc_per_epoch[str(epoch)] = train_acc
            if config['combinator'] == 'Linear':
                alpha_per_epoch[str(epoch)] = alpha_params
        print(f"Epoch {epoch} - "
              f"Accuracy: {round(100 * acc / config['len_train'], 3):.3f}% - "
              f"Time(epoch): {timedelta(seconds=time() - time1)} - "
              f"Time(tot): {timedelta(seconds=time() - time0)}")

    return train_acc_per_epoch, alpha_per_epoch


def test(config):
    test_acc_per_epoch = OrderedDict()
    load_dir = f'{config["save_dir"]}/weights/'
    model_list = sorted(list(map(lambda m: int(m.split('.')[0]), os.listdir(load_dir))))
    for epoch in model_list:
        model = load_dir + str(epoch) + '.pth'
        config['network'].load_state_dict(torch.load(model))
        Y_batch, Predicted = [], []
        with torch.no_grad():
            config['network'].eval()
            acc = 0
            for idx, (X_batch, y_batch) in enumerate(config['data_test']):
                X_batch = X_batch.to(config['device']).view(X_batch.shape[0], -1)
                y_batch = y_batch.to(config['device'])
                y_pred = config['network'](X_batch).to(config['device'])
                predicted = torch.argmax(y_pred.data, 1)
                acc += (predicted == y_batch).sum().item()
                Y_batch += y_batch
                Predicted += predicted
        test_acc = 100 * acc / config['len_test']
        test_acc_per_epoch[str(epoch)] = test_acc
        print(f'epoch: {epoch} - test accuracy: {round(test_acc, 3):.3f}')
    if config['combinator'] == 'Linear':
        shutil.rmtree(load_dir)
    return test_acc_per_epoch


# =====================================================================

def main(config):
    train_acc_per_epoch, test_acc_per_epoch, alpha_per_epoch = [], [], []

    if not test_only:
        print('...Training...')
        train_acc_per_epoch, alpha_per_epoch = train(config)

    print('...Testing...')
    test_acc_per_epoch = test(config)

    _ = save_results(config, train_acc_per_epoch, test_acc_per_epoch, alpha_per_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("-test_only", action="store_true")
    parser.add_argument("-save_test", action="store_true")
    args = parser.parse_args()
    test_only, save_test = args.test_only, args.save_test
    run_configs = load_run_config('run_config.json')
    np.random.seed(run_configs['random_seed'])
    torch.manual_seed(run_configs['random_seed'])
    configs = generate_configs(run_configs)
    time0 = time()
    for idx, conf in enumerate(configs):
        print(f'[{idx + 1}/{len(configs)}] {conf["save_dir"]}')
        try:
            main(conf)
        except Exception as e:
            raise
    print('End')
