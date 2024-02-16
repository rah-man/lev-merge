# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import numpy as np
import torch
import torch.nn as nn
from datasets import get_dataset
from torch.optim import SGD

from backbone.SimpleMLP import SimpleMLP
from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Learning without Forgetting.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Penalty weight.')
    parser.add_argument('--softmax_temp', type=float, default=2,
                        help='Temperature of the softmax function.')
    return parser


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new, temp):
    log_p = torch.log_softmax(new / temp, dim=1)
    q = torch.softmax(old / temp, dim=1)
    distil_loss = nn.functional.kl_div(log_p, q, reduction="batchmean")      
    return distil_loss


class LwfMod(ContinualModel):
    NAME = 'lwf_mod'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.dataset = get_dataset(args)
        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        nc = get_dataset(args).N_TASKS * self.cpt
        self.eye = torch.tril(torch.ones((nc, nc))).bool().to(self.device)

        self.cls2idx = {}
        self.idx2cls = {}
        self.previous_model = None

    def expand_model(self):
        base_weight = self.net.net[1][0].weight
        classifier_weight = self.net.net[1][2].weight

        out_features = self.current_task * self.dataset.N_CLASSES_PER_TASK
        model = SimpleMLP(input_size=768, output_size=out_features)

        model.net[1][0].weight.data = base_weight
        model.net[1][2].weight.data[:classifier_weight.size(0)] = classifier_weight
        self.net = model.to(self.device)

    def begin_task(self, dataset):
        """
        dataset is already in current dataset?
        """
        self.net.eval()

        if "mod" in self.args.dataset and hasattr(dataset, "update_classmap"):
            dataset.update_classmap()

        if self.current_task > 0:
            self.previous_model = copy.deepcopy(self.net)
            self.previous_model.eval()

            logits = []
            train_loader, _ = dataset.get_data_loaders(keep=False)

            for images, labels, not_aug in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                with torch.no_grad():
                    log = self.net(images).cpu()
                    logits.append(log)
            setattr(dataset.train_loader.dataset, 'logits', torch.cat(logits))
        self.net.train()
        self.current_task += 1            
    

    def observe(self, inputs, labels, not_aug_inputs, logits=None, i=0):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        # mask = self.eye[self.current_task * self.cpt - 1]
        # loss = self.loss(outputs[:, mask], labels)
        loss = self.loss(outputs, labels)
        
        if logits is not None:
            """
            as logits has been calculated for all current training data,
            to calculate the loss: take current length of outputs TIMES current loader iteration
            """
            # print("logits.size():", logits.size())
            # print("outputs.size():", outputs.size())

            current_size = outputs.size(0)
            previous_distribution = logits[i*current_size:(i+1)*current_size]
            current_distribution = outputs[:, :logits.size(1)]

            # print("prev_dist.size():", previous_distribution.size())
            # print("current_dist.size():", current_distribution.size())

            # print("prev_dist")
            # print(previous_distribution)

            # print("\ncurr_dist")
            # print(current_distribution)          

            # print(modified_kl_div(previous_distribution, current_distribution, self.args.softmax_temp))
            # exit()            
            loss += self.args.alpha * modified_kl_div(previous_distribution, 
                                                        current_distribution,
                                                        self.args.softmax_temp)

        loss.backward()
        self.opt.step()

        return loss.item()
