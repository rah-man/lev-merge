from torch import nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """Simple MLP with one hidden layer."""

    def __init__(self, in_features=768, num_classes=10, out_features=1000, **kwargs):
        super().__init__()
        # main part of the network
        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)
        # self.fcx = nn.Linear(in_features=out_features, out_features=out_features)
        # last classifier layer (head) with as many outputs as classes
        self.fc2 = nn.Linear(in_features=out_features, out_features=num_classes)

        # and `head_var` with the name of the head, so it can be removed when doing incremental learning experiments
        self.head_var = 'fc2'

    def forward(self, x):
        out = self.fc1(x.float())
        out = F.relu(out)
        out = self.fc2(out)
        # out = self.fc2(F.relu(self.fcx(out)))
        return out


def simpleMLP(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    return SimpleMLP(**kwargs)
