import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from avalanche.benchmarks.utils import AvalancheTensorDataset
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.models import SimpleMLP
from avalanche.training import JointTraining

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

classes = [100, 200, 1011]
bsize = [64, 10, 256]
train_path = ["../cifar100_coarse_train_embedding_nn.pt", "../ds_cub_train.pt", "/data/shared/inat_train.pt"]
test_path = ["../cifar100_coarse_test_embedding_nn.pt", "../ds_cub_test.pt", "/data/shared/inat_test.pt"]
train_embedding_path = train_path[d]
test_embedding_path = test_path[d]
num_classes = classes[d]
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
        
train_x = torch.stack(train_x)
train_y = torch.tensor(train_y)

test_x = torch.stack(test_x)
test_y = torch.tensor(test_y)

train_dataset = AvalancheTensorDataset(train_x, train_y)
test_dataset = AvalancheTensorDataset(test_x, test_y)

scenario = nc_benchmark(
    train_dataset, test_dataset, n_experiences=n_task, shuffle=True,
    task_labels=False
)

train_stream = scenario.train_stream
test_stream = scenario.test_stream

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleMLP(num_classes=num_classes, input_size=768, hidden_size=1000, hidden_layers=1)
optimiser = optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
criterion = nn.CrossEntropyLoss()
print(model)

strategy = JointTraining(model, optimiser, criterion, 
                        train_mb_size=batch,
                        train_epochs=100,
                        eval_mb_size=batch,
                        device=device,)

results = []
strategy.train(train_stream)
results.append(strategy.eval(test_stream))
print(results)

pickle.dump(results, open(pickle_file, "wb"))