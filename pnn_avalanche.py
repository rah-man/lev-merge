import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from avalanche.benchmarks.utils import AvalancheTensorDataset
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, \
    accuracy_metrics, loss_metrics, timing_metrics
from avalanche.logging import InteractiveLogger    
from avalanche.models import PNN
from avalanche.training import PNNStrategy
from avalanche.training.plugins import EvaluationPlugin

parser = argparse.ArgumentParser()
parser.add_argument("-d")
parser.add_argument("-t", "--task")
parser.add_argument("-lr", "--learning_rate")
parser.add_argument("-wd", "--weight_decay")
parser.add_argument("-mom", "--momentum")
parser.add_argument("-p", "--pickle")
args = parser.parse_args()

d = int(args.d)
n_task = int(args.task)
lr = float(args.learning_rate)
wd = float(args.weight_decay)
mom = float(args.momentum)
pickle_file = args.pickle

train_path = ["../cifar100_coarse_train_embedding_nn.pt", "../ds_cub_train.pt", "/data/shared/inat_train.pt"]
test_path = ["../cifar100_coarse_test_embedding_nn.pt", "../ds_cub_test.pt", "/data/shared/inat_train.pt"]
train_embedding_path = train_path[d]
test_embedding_path = test_path[d]

classes = [100, 200, 1011]
bsize = [64, 10, 256]
batch = bsize[d]

train_set = torch.load(open(train_embedding_path, "rb"))
test_set = torch.load(open(test_embedding_path, "rb"))

train_x, train_y = [], []
test_x, test_y = [], []

if d == 0:
    # cifar100 coarse has 3 elements: x, y_coarse, y_fine
    for image, coarse_label, label in train_set["data"]:
        train_x.append(image)
        train_y.append(label)
        
    for image, coarse_label, label in test_set["data"]:
        test_x.append(image)
        test_y.append(label)        
else:
    # cub and inaturalist have 2 elements: x, y
    for image, label in train_set["data"]:
        train_x.append(image)
        train_y.append(label)
        
    for image, label in test_set["data"]:
        test_x.append(image)
        test_y.append(label)
        
if d != 2:
    per_exp = None
elif d == 2:
    if n_task == 10:
        per_exp = {0: 102}
    elif n_task == 20:
        per_exp = {0: 61}            
        
train_x = torch.stack(train_x)
train_y = torch.tensor(train_y)

test_x = torch.stack(test_x)
test_y = torch.tensor(test_y)

train_dataset = AvalancheTensorDataset(train_x, train_y)
test_dataset = AvalancheTensorDataset(test_x, test_y)

scenario = nc_benchmark(
    train_dataset, test_dataset, n_experiences=n_task, shuffle=True,
    task_labels=False, per_exp_classes=per_exp
)

train_stream = scenario.train_stream
test_stream = scenario.test_stream

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PNN(in_features=768, hidden_features_per_column=1000)
optimiser = optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=30)
criterion = nn.CrossEntropyLoss()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[InteractiveLogger()],
    benchmark=scenario,
    strict_checks=False
)

strategy = PNNStrategy(model, optimiser, criterion, 
                        train_mb_size=batch, 
                        train_epochs=100, 
                        eval_mb_size=batch,
                        device=device,
                        evaluator=eval_plugin,
                        plugins=[scheduler],)

results = []
for experience in train_stream:
    print(model)
    res = strategy.train(experience)
    results.append(strategy.eval(test_stream))

print(results)
pickle.dump(results, open(pickle_file, "wb"))