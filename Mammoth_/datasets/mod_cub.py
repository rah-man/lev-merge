# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as transforms

from backbone.SimpleMLP import SimpleMLP
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
# from backbone.ResNet18 import resnet18

from base import get_data

from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders

class BaseDataset(Dataset):
    def __init__(self, x, y, transform=None, target_transform=None):
        self.data = x
        self.targets = torch.tensor(y)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            # target_transform is a dictionary of cls2idx
            y = self.target_transform[y.item()]
        
        return x, y

class XDataset(BaseDataset):
    def __init__(self, x, y, transform=None, target_transform=None):
        super().__init__(x, y, transform, target_transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        not_aug_img = x

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return x, y, not_aug_img

class ModCUB200(ContinualDataset):
    NAME = 'mod-cub200'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10

    def __init__(self, args):
        super().__init__(args)

        self.N_TASKS = args.ntask
        self.N_CLASSES_PER_TASK = 200 // self.N_TASKS   # cub has 200 classes
        self.cls2idx = {}
        self.idx2cls = {}

        # HOW CAN THIS PATH BE USED HERE?
        # GOT IT
        # IT ASSUMES PWD STARTS FROM "mammoth/" directory so it gets out one level and find the dataset
        # even though this particular file is inside mammoth/datasets/ directory
        train_embedding_path, test_embedding_path = "../ds_cub_train.pt", "../ds_cub_test.pt"
        self.data, self.class_order = get_data(
            train_embedding_path, 
            test_embedding_path, 
            num_tasks=self.N_TASKS,
            seed=None)    

    # print(class_order)
    # print(data.keys())
    # print(data[0].keys())
    # print(len(data[0]["train"]["x"]))
    # print(len(data[0]["train"]["y"]))
    # print(len(data[0]["val"]["x"]))
    # print(len(data[0]["val"]["x"]))
    # print(len(data[0]["test"]["x"]))
    # print(len(data[0]["test"]["x"]))
    # print(data[0]["classes"])

    def update_classmap(self):
        # print("\tupdate_classmap: VALUE OF SELF.I:", self.i)
        new_cls = self.data[self.i]["classes"]
        cls_ = list(self.cls2idx.keys())
        cls_.extend(new_cls)
        self.cls2idx = {v: k for k, v in enumerate(cls_)}
        self.idx2cls = {k: v for k, v in enumerate(cls_)}    

    def get_data_loaders(self, keep=True):
        # print("\tget_data_loaders: VALUE OF SELF.I:", self.i)
        current_data = self.data[self.i]
        # print(f"\tModCIFAR100 current_data_targets: {np.unique(current_data['train']['y']).tolist()}")
        train_x, train_y = current_data["train"]["x"], current_data["train"]["y"]
        test_x, test_y = current_data["val"]["x"], current_data["val"]["y"]
        
        train_dataset = XDataset(train_x, train_y, target_transform=self.cls2idx)
        test_dataset = BaseDataset(test_x, test_y, target_transform=self.cls2idx)

        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
        if keep:
            self.test_loaders.append(test_loader)
        # print("\t>>> LEN(TEST_LOADERS):", len(self.test_loaders))
        return self.train_loader, test_loader

    @staticmethod
    def get_transform():
        return transforms.Compose([nn.Identity()])

    @staticmethod
    def get_backbone():
        return SimpleMLP(input_size=768, output_size=200)
        # return resnet18(ModCIFAR100.N_CLASSES_PER_TASK
        #                 * ModCIFAR100.N_TASKS)

    def get_incremental_backbone(self):
        """
        not static as N_CLASS_PER_TASK is specific 
        initialise with outpout_size=N_CLASSES_PER_TASK
        """
        return SimpleMLP(input_size=768, output_size=self.N_CLASSES_PER_TASK)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Compose([nn.Identity()])

    @staticmethod
    def get_denormalization_transform():
        return transforms.Compose([nn.Identity()])

    @staticmethod
    def get_epochs():
        return 5

    @staticmethod
    def get_batch_size():
        return 10

    @staticmethod
    def get_minibatch_size():
        return ModCUB200.get_batch_size()

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.SGD(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [35, 45], gamma=0.1, verbose=False)
        scheduler = torch.optim.lr_scheduler.StepLR(model.opt, step_size=30)
        return scheduler

