import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

"""
New addition:
    - add bias layer to the expert, not just outside the expert.
    - check the performance
"""

class BiasLayer(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        # Initialize alpha and beta with requires_grad=False and only set to True during Stage 2
        self.alpha = torch.nn.Parameter(torch.ones(1), requires_grad=requires_grad)
        self.beta = torch.nn.Parameter(torch.zeros(1), requires_grad=requires_grad)

    def forward(self, x):
        return self.alpha * x + self.beta

class Expert(nn.Module):
    def __init__(self, input_size=768, hidden_size=100, output_size=2, projected_output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc3 = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.bias = BiasLayer()
        self.mapper = nn.Linear(in_features=output_size, out_features=projected_output_size, bias=False)
        self.normalise1 = nn.InstanceNorm1d(num_features=hidden_size)
        self.normalise2 = nn.InstanceNorm1d(num_features=hidden_size)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = F.relu(self.normalise1(self.fc1(x)))
        out = F.relu(self.normalise2(self.fc2(out)))
        out = self.mapper(self.fc3(out))
        return out
    
    def bias_forward(self, logits):
        return self.bias(logits)

class DynamicExpert(nn.Module):
    def __init__(self, input_size=768, hidden_size=100, total_cls=100):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.total_cls = total_cls
        self.gate = None
        self.experts = None
        self.bias_layers = None
        self.prev_classes = []
        self.cum_classes = set()
        self.relu = nn.ReLU()

    def expand_expert(self, seen_cls, new_cls):
        self.seen_cls = seen_cls
        self.new_cls = new_cls

        if not self.experts:
            self.prev_classes.append(self.new_cls)
            gate = nn.Linear(in_features=self.input_size, out_features=1)
            experts = nn.ModuleList([Expert(input_size=self.input_size, hidden_size=self.hidden_size, output_size=new_cls, projected_output_size=new_cls)])
            self.bias_layers = nn.ModuleList([BiasLayer()])
            self.num_experts = len(experts)            
        else:            
            self.prev_classes.append(self.new_cls)
            gate = nn.Linear(in_features=self.input_size, out_features=self.num_experts+1)                  
            experts = copy.deepcopy(self.experts)
            experts.append(Expert(input_size=self.input_size, hidden_size=self.hidden_size, output_size=new_cls, projected_output_size=new_cls))
            self.num_experts = len(experts)
            for expert_index, module in enumerate(experts):
                start = sum(self.prev_classes[:expert_index])
                end = start + self.prev_classes[expert_index]

                weight = module.mapper.weight
                input_size = module.mapper.in_features
                new_mapper = nn.Linear(in_features=input_size, out_features=sum(self.prev_classes), bias=False)

                with torch.no_grad():
                    all_ = {i for i in range(sum(self.prev_classes))}
                    kept_ = {i for i in range(start, end)}
                    removed_ = all_ - kept_
                    
                    upper_bound = sum(self.prev_classes[:expert_index+1])

                    new_mapper.weight[start:end, :] = weight if weight.size(0) <= new_cls else weight[start:upper_bound, :]
                    new_mapper.weight[list(removed_)] = 0.
                    module.mapper = new_mapper

            self.bias_layers.append(BiasLayer())
        
        self.gate = gate
        self.experts = experts

    def calculate_gate_norm(self):
        w1 = nn.utils.weight_norm(self.gate, name="weight")
        print(w1.weight_g)
        nn.utils.remove_weight_norm(w1)

    def bias_forward(self, task, output):
        """Modified version from FACIL"""
        return self.bias_layers[task](output)

    def freeze_previous_experts(self):
        for i in range(len(self.experts) - 1):
            e = self.experts[i]
            for param in e.parameters():
                param.requires_grad = False

    def freeze_all_experts(self):
        for e in self.experts:
            for param in e.parameters():
                param.requires_grad = False

    def freeze_all_bias(self):
        for b in self.bias_layers:
            b.alpha.requires_grad = False
            b.beta.requires_grad = False

    def unfreeze_all_bias(self):
        for b in self.bias_layers:
            b.alpha.requires_grad = True
            b.beta.requires_grad = True

    def set_gate(self, grad):
        for name, param in self.named_parameters():
            if name == "gate":
                param.requires_grad = grad

    def unfreeze_all(self):
        for e in self.experts:
            for param in e.parameters():
                param.requires_grad = True

    def forward(self, x, task=None, train_step=2):
        gate_outputs = None
        if train_step == 1:            
            expert_outputs = self.experts[task](x)
        else:
            gate_outputs = self.gate(x)
            
            gate_outputs_uns = torch.unsqueeze(gate_outputs, 1)
            
            expert_outputs = [self.experts[i](x) for i in range(self.num_experts)]
            expert_outputs = torch.stack(expert_outputs, 1)
            expert_outputs = gate_outputs_uns@expert_outputs
            expert_outputs = torch.squeeze(expert_outputs, 1) # only squeeze the middle 1 dimension

        return expert_outputs, gate_outputs

    def predict(self, x, task):
        expert_output = self.experts[task](x)
        return expert_output
        