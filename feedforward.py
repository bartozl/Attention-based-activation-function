import torch
import torch.nn.functional as F
import os
import json
import argparse
from time import time
from datetime import timedelta
import utils
from pathlib import Path


def train(config):
    print('number of trained parameters', sum(p.numel() for p in config['network'].parameters() if p.requires_grad))
    load_this_model = ''
    for epoch in range(1, config['epochs'] + 1):
        dest_path = f'{config["save_dir"]}/weights/{epoch}.pth'
        full_dest_path = Path(str(Path().absolute()) + "/" + dest_path)
        if Path.exists(full_dest_path):
            load_this_model = full_dest_path
            continue
        if load_this_model != '':
            config['network'].load_state_dict(torch.load(load_this_model))
        time1 = time()  # time per epoch
        config['network'].train()  # train mode
        acc, loss = 0, 0
        for idx, (X_batch, y_batch) in enumerate(config['data_train']):
            config['optimizer'].zero_grad()
            X_batch = X_batch.to(config['device']).view(X_batch.shape[0], -1)
            y_batch = y_batch.to(config['device'])
            y_pred, _, _, params = config['network'](X_batch)
            loss = F.nll_loss(y_pred, y_batch)
            if config['lambda_l1'] != 0:
                reg_loss = F.l1_loss(params, target=torch.zeros_like(params), reduction='sum')
                loss += reg_loss * config['lambda_l1']
            loss.backward()
            config['optimizer'].step()
            predicted = torch.argmax(y_pred.data, 1)
            acc += (predicted == y_batch).sum().item()
        if epoch % config['save_every'] == 0:
            # train_acc_per_epoch[str(epoch)] = utils.save_state(config, dest_path, acc)
            acc = utils.save_state(config, dest_path, acc)
            utils.save_results(config, acc, None, epoch, loss.item())
        print(f"Epoch {epoch} - "
              f"Accuracy: {round(acc, 3):.3f}% - "
              f"Loss: {loss:.3f} - "
              f"Time(epoch): {timedelta(seconds=time() - time1)} - "
              f"Time(tot): {timedelta(seconds=time() - time0)}")


def test(config):
    load_dir = f'{config["save_dir"]}/weights/'
    model_list = sorted(list(map(lambda m: int(m.split('.')[0]), os.listdir(load_dir))))
    if hr_test is None:
        with open(f'{config["save_dir"]}/results.json') as f:
            results = json.load(f)
            start_from_epoch = list(results["test_acc"])[-1] if results["test_acc"]['1'] is not None else 0
        model_list = model_list[int(start_from_epoch):]

    for epoch in model_list:
        # print(epoch)
        acc = 0
        model = f'{load_dir}{epoch}.pth'
        state_dict = torch.load(model)
        config['network'].load_state_dict(state_dict)
        config['network'].eval()
        Y_batch, Predicted = [], []
        with torch.no_grad():
            for idx, (X_batch, y_batch) in enumerate(config['data_test']):
                X_batch = X_batch.to(config['device']).view(X_batch.shape[0], -1)
                y_batch = y_batch.to(config['device'])
                y_pred, _, _, _ = config['network'](X_batch)
                predicted = torch.argmax(y_pred.data, 1)
                acc += (predicted == y_batch).sum().item()
                Y_batch += y_batch
                Predicted += predicted
        test_acc = 100 * acc / config['len_test']
        # test_acc_per_epoch[str(epoch)] = test_acc
        utils.save_results(config, None, test_acc, str(epoch), hr_test=hr_test)
        print(f'epoch: {epoch} - test accuracy: {round(test_acc, 3):.3f}')


# =====================================================================

def main(config):
    train_acc_per_epoch, test_acc_per_epoch = [], []  # store accuracies history

    print(f'seed: {torch.initial_seed()}, dataset: {config["dataset"]}, '
          f'len_train: {config["len_train"]}, len_test:{config["len_test"]}')
    print('...Training...')
    train(config)

    print('...Testing...')
    test(config)

    # _ = utils.save_results(config, train_acc_per_epoch, test_acc_per_epoch, test_only, save_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("-hr_test", type=float, default=None)
    parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-colab", action="store_true")
    args = parser.parse_args()
    hr_test = args.hr_test
    # assert hr_test in [None, "1", "2"], "hr_test must be 1, 2 or None!!!"
    run_configs = utils.load_run_config(args.config)
    configs = utils.generate_configs(run_configs, hr_test, args.colab)  # list of configurations (=dict) to be trained
    time0 = time()  # total run time
    for i, conf in enumerate(configs):
        print(torch.cuda.current_device(), torch.cuda.get_device_name(0), torch.cuda.is_available())
        print(f'[{i + 1}/{len(configs)}] {conf["save_dir"]}')
        try:
            utils.reset_seed(conf['random_seed'])
            main(conf)
        except Exception as e:
            raise
    print('End')
