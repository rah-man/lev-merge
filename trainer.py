"""
Problem setting:
Assuming there are 100 classes

RUN CUB with 10 and 20 classes following the CIFAR-100


1. N classes per task, N = 10
2. various numbers of classes per task, varying from 5 to 15
3. 50 + N classes per task, N = 1 or 2
"""


"""
When T1 comes
1. create prototype per class in T1
    a. mean for clustering
    b. GMM for generating pseudo samples (n = 1..3)
2. extension steps (start from T2)
    a. calculate similarity of each prototype with existing experts' prototypes
    b. if one prototype is similar to one expert, then we extend the expert and update the expert with new training samples with REAL REPLAY SAMPLES (not pseudo researhal?)
    c. for all remaining prototypes, cluster them and create an expert for each cluster
"""

import argparse
import copy
import itertools
import numpy as np
import pickle
import random
import statistics
import time
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.optim as optim
import torchvision.models as models
import sys

from collections import Counter
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score, ConfusionMatrixDisplay, confusion_matrix, classification_report, f1_score
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor

# own library
import clusterlib

from base import BaseDataset, get_data, get_cifar100_coarse
from earlystopping import EarlyStopping
from expert import DynamicExpert
from mets import Metrics2
from replay import RandomReplay

class Trainer:
    def __init__(self, criterion, dataset, lr, class_order, epochs=100, device="cpu", n_task=10, metric=None):
        self.criterion = criterion
        self.dataset = dataset
        self.lr = lr
        self.class_order = class_order
        self.epochs = epochs
        self.device = device
        self.n_task = n_task
        self.metric = metric
        
        self.seen_cls = 0
        self.cls2idx = {}
        self.idx2cls = {}
        
        self.model = DynamicExpert()

    def get_numpy_from_dataset(self, task, type_="trn", transform=False):
        current_dataset = self.dataset[task]
        
        x = torch.vstack(current_dataset[type_]["x"]).numpy()
        y = torch.tensor(current_dataset[type_]["y"]).numpy()
        
        if transform:
            y = np.vectorize(self.cls2idx.get)(y)
        return x, y
        
    def train(self):
        # run continual learning with clustering here
        current_task = 0
        train_loaders, val_loaders = [], []
        train_dataset, val_dataset = [], []
        
        for task in range(self.n_task):
            # update class map
            pass
    
    def test(self):
        # run testing from the stored dataset
        pass
    
if __name__ == "__main__":        
    parser = argparse.ArgumentParser()
    parser.add_argument("-dtype", type=int)
    parser.add_argument("-batch", type=int)
    parser.add_argument("-memory", type=int)
    parser.add_argument("-epochs", type=int)
    parser.add_argument("-ntask", type=int)

    args = parser.parse_args()

    # dtype = 0/cifar100-coarse & 1/cub
    dtype = args.dtype
    batch = args.batch
    memory = args.memory
    epochs = args.epochs
    n_task = args.ntask

    train_path = ["cifar100_coarse_train_embedding_nn.pt", "ds_cub_train.pt"]
    test_path = ["cifar100_coarse_test_embedding_nn.pt", "ds_cub_test.pt"]
    
    train_embedding_path = train_path[dtype]
    val_embedding_path = None
    test_embedding_path = test_path[dtype]

    n_class = 100 if dtype == 0 else 200
    lr = 0.01
    random_replay = RandomReplay(mem_size=memory)
    criterion = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype == 0:
        # cifar100 coarse
        data, task_cla, class_order = get_cifar100_coarse(train_embedding_path, test_embedding_path)
    else:
        data, task_cla, class_order = get_data(train_embedding_path, test_embedding_path, validation=0.2, num_tasks=n_task, expert=True,)
    
    metric = Metrics2()    
    trainer = Trainer(criterion, data, lr, class_order, epochs=epochs, device=device, n_task=n_task, metric=metric)

    walltime_start, processtime_start = time.time(), time.process_time()
    _, _ = trainer.train()
    walltime_end, processtime_end = time.time(), time.process_time()
    elapsed_walltime = walltime_end - walltime_start
    elapsed_processtime = processtime_end - processtime_start
    
    print('Execution time:', )
    print(f"CPU time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_processtime))}\tWall time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_walltime))}")
    print(f"CPU time: {elapsed_processtime}\tWall time: {elapsed_walltime}")

    faa = trainer.metric.final_average_accuracy()
    ff = trainer.metric.final_forgetting()
    print(f"FAA: {faa}")
    print(f"FF: {ff}")
    print("\nTRAINER.METRIC.ACCURACY")
    for k, v in trainer.metric.accuracy.items():
        print(f"{k}: {v}")
    print()
    
    if test_embedding_path:
        test_result = trainer.test()