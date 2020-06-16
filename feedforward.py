import numpy as np
import torch
import torch.nn.functional as F
import os
from collections import OrderedDict
import argparse
from time import time
from datetime import timedelta
import utils


def train(config):
    train_acc_per_epoch = OrderedDict()  # store train accuracy history
    print('number of trained parameters', sum(p.numel() for p in config['network'].parameters() if p.requires_grad))
    for epoch in range(1, config['epochs'] + 1):
        time1 = time()  # time per epoch
        config['network'].train()  # train mode
        acc = 0
        for idx, (X_batch, y_batch) in enumerate(config['data_train']):
            config['optimizer'].zero_grad()
            X_batch = X_batch.to(config['device']).view(X_batch.shape[0], -1)
            y_batch = y_batch.to(config['device'])
            y_pred = config['network'](X_batch).to(config['device'])
            loss = F.nll_loss(y_pred, y_batch)
            '''
            if config['lambda_l1'] != 0:
                reg_loss = 0
                for name, param in config['network'].named_parameters():
                    print(name)
                    l1_loss = F.l1_loss(param, target=torch.zeros_like(param), size_average=False)
                    reg_loss += l1_loss
                loss += config['lambda_l1'] * reg_loss
            '''
            if config['lambda_l1'] != 0:
                for name, buf in config['network'].named_buffers():
                    if name == '1.alpha':
                        l1_loss = F.l1_loss(buf, target=torch.zeros_like(buf), reduction='sum')
                        loss += l1_loss * config['lambda_l1']
            loss.backward()
            config['optimizer'].step()
            predicted = torch.argmax(y_pred.data, 1)
            acc += (predicted == y_batch).sum().item()
        if epoch % config['save_every'] == 0:
            train_acc_per_epoch[str(epoch)] = utils.save_state(config, epoch, acc)
        print(f"Epoch {epoch} - "
              f"Accuracy: {round(100 * acc / config['len_train'], 3):.3f}% - "
              f"Time(epoch): {timedelta(seconds=time() - time1)} - "
              f"Time(tot): {timedelta(seconds=time() - time0)}")

    return train_acc_per_epoch


def test(config):
    test_acc_per_epoch = OrderedDict()
    load_dir = f'{config["save_dir"]}/weights/'
    model_list = sorted(list(map(lambda m: int(m.split('.')[0]), os.listdir(load_dir))))
    for epoch in model_list:
        model = load_dir + str(epoch) + '.pth'
        state_dict = torch.load(model)
        config['network'].load_state_dict(state_dict)
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

    return test_acc_per_epoch


# =====================================================================

def main(config):
    train_acc_per_epoch, test_acc_per_epoch = [], []  # store accuracies history

    if not test_only:
        print('...Training...')
        train_acc_per_epoch = train(config)

    print('...Testing...')
    test_acc_per_epoch = test(config)

    _ = utils.save_results(config, train_acc_per_epoch, test_acc_per_epoch, test_only, save_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("-test_only", action="store_true")
    parser.add_argument("-save_test", action="store_true")
    parser.add_argument("-colab", action="store_true")
    args = parser.parse_args()
    test_only, save_test = args.test_only, args.save_test
    run_configs = utils.load_run_config('run_config.json')
    np.random.seed(run_configs['random_seed'])  # allows reproducibility
    torch.manual_seed(run_configs['random_seed'])  # allows reproducibility
    configs = utils.generate_configs(run_configs, test_only, args.colab)  # list of configurations (=dict) to be trained
    time0 = time()  # total run time
    for i, conf in enumerate(configs):
        print(f'[{i + 1}/{len(configs)}] {conf["save_dir"]}')
        try:
            main(conf)
        except Exception as e:
            raise
    print('End')
