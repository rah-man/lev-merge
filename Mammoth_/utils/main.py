# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np # needed (don't change it)
import importlib
import os
import socket
import sys
import pickle
import time


mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets') # mod_har.py
sys.path.append(mammoth_path + '/backbone') # SimpleMLP.py
sys.path.append(mammoth_path + '/models')   # lwf_har.py etc

import datetime
import uuid
from argparse import ArgumentParser

import setproctitle
import torch
from datasets import NAMES as DATASET_NAMES
from datasets import ContinualDataset, get_dataset
from models import get_all_models, get_model, lwf_har, ewc_har, gdumb_har, der_har

from utils.args import add_management_args
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils.continual_training import train as ctrain
from utils.distributed import make_dp
import utils.training as training
import utils.training_mod as training_mod

from sklearn.metrics import f1_score, classification_report

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')    
    torch.set_num_threads(4)
    add_management_args(parser)
    args = parser.parse_known_args()[0]

    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(mod, 'Buffer'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args


def main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    dataset = get_dataset(args)

    # print(dataset)
    # print(dataset.__dict__)
    # exit()

    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()        
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and args.minibatch_size is None:
        args.minibatch_size = dataset.get_minibatch_size()

    if args.backbone_type is None and "mod" in args.dataset:
        args.backbone_type = "incremental"
        backbone = dataset.get_incremental_backbone()
    elif "mod" in args.dataset:
        backbone = dataset.get_backbone()
    else:
        backbone = dataset.get_backbone()
        
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    # print(f"args.n_epochs: {args.n_epochs}")
    # print(f"args.batch_size: {args.batch_size}")
    # print(f"args.minibatch_size: {args.minibatch_size}")
    # print(f"backbone: {backbone}")
    # print(f"model: {model}")
    # exit()

    # print("===========")
    # print(args)
    # print("===========")
    # exit()

    if args.distributed == 'dp':
        model.net = make_dp(model.net)
        model.to('cuda:0')
        args.conf_ngpus = torch.cuda.device_count()
    elif args.distributed == 'ddp':
        # DDP breaks the buffer, it has to be synchronized.
        raise NotImplementedError('Distributed Data Parallel not supported yet.')

    if args.debug_mode:
        args.nowand = 1

    model.device = "cuda" if torch.cuda.is_available() else "cpu"

    # set job name
    setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    train_start = time.process_time()
    if "mod" in args.dataset:
        class_il_acc, task_il_acc, task_train_time, task_params, epoch_detail = training_mod.train(model, dataset, args)
    elif isinstance(dataset, ContinualDataset):
        training.train(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)    
    train_end = time.process_time()

    # for LwF HAR
    # if isinstance(model, lwf_har.LwfHAR) or isinstance(model, ewc_har.EwcHAR) or isinstance(model, gdumb_har.GDumbHAR):    
    if 'har' in args.dataset:
        test_x, test_y = [], []

        if args.load_data:
            for i in range(len(dataset.data) - 1):
                dt = dataset.data[i]
                test_x.extend(dt['tst']['x'])
                test_y.extend(dt['tst']['y'])
        else:
            for inputs, labels in dataset.test_set["data"]:
                test_x.append(inputs)
                test_y.append(labels)

        if not args.load_data:
            test_x = torch.stack(test_x).to(model.device)
            test_y = torch.stack(test_y).to(model.device)
        else:
            test_x = torch.tensor(np.array(test_x)).to(model.device)
            test_y = torch.tensor(test_y)

        if not args.load_data:
            y = torch.tensor(np.vectorize(dataset.cls2idx.get)(test_y.cpu().numpy())).type(torch.int)
        else:
            y = test_y
        
        model.net.eval()
        with torch.no_grad():
            outputs = model.net(test_x)
        preds = torch.argmax(outputs, 1).cpu().numpy()
        labels = y.cpu().numpy()
        print(f"f1_micro: {100 * f1_score(labels, preds, average='micro')}")
        print(f"f1_macro: {100 * f1_score(labels, preds, average='macro')}")
        print(classification_report(labels, preds, zero_division=0))

        """ get prediction time on one test sample """  
        x_ = torch.unsqueeze(test_x[-1], 0)
        pred_start = time.process_time()
        with torch.no_grad():
            output = model.net(x_)
        pred = torch.argmax(output, 1).cpu().numpy()
        pred_end = time.process_time()

        # if hasattr(model, 'buffer'):
        #     print(model.buffer.buffer_size)

    total_params = sum(param.numel() for param in model.net.parameters())

    to_save = {
        # 'test_confusion_matrix': {'true': labels, 'preds': preds},
        'train_time': train_end - train_start,
        'class_il_acc': class_il_acc,
        'task_il_acc': task_il_acc,
        'task_train_time': task_train_time,
        # 'prediction_time': pred_end - pred_start,
        'total_parameters': total_params,
        'task_parameters': task_params,
        # 'data': dataset.data,
        'clsorder': dataset.class_order,
        'epoch_detail': epoch_detail,
    }

    # print(to_save)

    # print(f"task_train_time: {task_train_time}")
    # print(f"prediction_time: {pred_end - pred_start}")

    if args.save_path:
        pickle.dump(to_save, open(args.save_path, 'wb'))


    print(f'Parameters: {total_params}')
    # print(f'Task parameters: {task_params}')


if __name__ == '__main__':
    main()
