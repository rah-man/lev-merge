# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
import sys
import time
from argparse import Namespace
from typing import Tuple

import numpy as np
import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.metrics import forgetting
from utils.status import ProgressBar

try:
    import wandb
except ImportError:
    wandb = None

def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        # print("\tk in test_loader:", k)
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    print(args)

    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        # bypass this by setting args.wandb_entity="frahman" --> my wandb account
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
        # wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), mode="offline")
        args.wandb_url = None


    model.net.to(model.device)
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    # progress_bar = ProgressBar(verbose=not args.non_verbose)
    progress_bar = ProgressBar(verbose=False)

    # comment these lines below for irrelevant actions
    if not args.ignore_other_metrics:
        dataset_copy = get_dataset(args)
        for t in range(dataset.N_TASKS):
            model.net.train()
            _, _ = dataset_copy.get_data_loaders()
        if 'icarl' not in model.NAME and model.NAME != 'pnn':
            random_results_class, random_results_task = evaluate(model, dataset_copy)
        # print()

    # print(file=sys.stderr)
    train_time = {}
    task_params = {}
    epoch_detail = {}
    for t in range(dataset.N_TASKS):
        print("====TRAINING TASK:", t)
        model.net.train()     
        logits = None   
        if hasattr(model, 'begin_task'):
            """
            LWF should create a copy of old model here and run on current dataset
                and store the result in logits
            """
            model.begin_task(dataset)
            if dataset.train_loader:
                if hasattr(dataset.train_loader.dataset, "logits"):
                    logits = dataset.train_loader.dataset.logits
            # print("DONE UPDATING CLASSMAP")
        
        if t and not args.ignore_other_metrics:
            """
            LWF should evaluate
            """            
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]

        if "mod" in args.dataset and hasattr(model, "expand_model") and t > 0:
            """
            expand current model if t > 0
            """
            model.expand_model()
            # print(model)
            # print("\tDONE MODEL EXPANSION")

        print(model.net)

        train_loader, test_loader = dataset.get_data_loaders()
        scheduler = dataset.get_scheduler(model, args)
        
        # store model parameters
        total_params = sum(param.numel() for param in model.net.parameters())
        task_params[t] = total_params

        epoch_acc, epoch_loss = [], []

        train_start = time.process_time()        
        for epoch in range(model.args.n_epochs):
            if args.model == 'joint':
                continue
            for i, data in enumerate(train_loader):
                if args.debug_mode and i > 3:
                    break
                if logits is not None:
                # if hasattr(dataset.train_loader.dataset, 'logits'):
                    # print("I CAN HAS LOGITS?")
                    inputs, labels, not_aug_inputs = data
                    inputs = inputs.to(model.device)
                    labels = labels.type(torch.LongTensor).to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs, logits, i)
                else:
                    inputs, labels, not_aug_inputs = data
                    # print(f"\tNOT-LOGITS: inputs.size(): {inputs.size()}\tlabels.size(): {labels.size()}\tnot_aug_inputs.size(): {not_aug_inputs.size()}")
                    inputs, labels = inputs.to(model.device), labels.type(torch.LongTensor).to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.meta_observe(inputs, labels, not_aug_inputs)
                # assert not math.isnan(loss)

                progress_bar.prog(i, len(train_loader), epoch, t, loss)

            
            ep_acc = model.calculate_accuracy(train_loader)
            epoch_acc.append(ep_acc)
            epoch_loss.append(loss)

            if scheduler is not None:
                scheduler.step()

        train_end = time.process_time()
        train_time[t] = train_end - train_start

        # ADD TASK EPOCH TO LOGGER
        epoch_detail[t] = {'epoch_accuracy': epoch_acc, 'epoch_loss': epoch_loss}

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        # print("\tEVALUATE_AFTER_TRAINING")
        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        if not args.disable_log:
            logger.log(mean_acc)
            logger.log_fullacc(accs)

        if not args.nowand:
            d2={'RESULT_class_mean_accs': mean_acc[0], 'RESULT_task_mean_accs': mean_acc[1],
                **{f'RESULT_class_acc_{i}': a for i, a in enumerate(accs[0])},
                **{f'RESULT_task_acc_{i}': a for i, a in enumerate(accs[1])}}

            wandb.log(d2)

        dataset.i += 1
        
    print(f"CLASS_IL_ACC: \n\t{results[-1]}")
    print(f"TASK_IL_ACC: \n\t{results_mask_classes[-1]}")
    print(f"Final_acc: {np.mean(results[-1])}")
    print(f"Forgetting: {forgetting(results)}")

    # print("\tLEN(RESULTS):", len(results))
    # print("\tLEN(RESULTS_MASK_CLASSES):", len(results_mask_classes))
    # for res in results:
    #     print(res)
    # print()
    # for res in results_mask_classes:
    #     print(res)
    if not args.disable_log and not args.ignore_other_metrics:
        logger.add_bwt(results, results_mask_classes)
        logger.add_forgetting(results, results_mask_classes)
        if 'icarl' not in model.NAME and model.NAME != 'pnn':
            logger.add_fwt(results, random_results_class,
                    results_mask_classes, random_results_task)

    if not args.disable_log:
        logger.write(vars(args))
        if not args.nowand:
            d = logger.dump()
            d['wandb_url'] = wandb.run.get_url()
            wandb.log(d)

    if not args.nowand:
        wandb.finish()

    return results, results_mask_classes, train_time, task_params, epoch_detail
