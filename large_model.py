import torch
import torch.nn as nn
import os

def adjust_list_size(lst, target_size):
    if len(lst) < target_size:
        num_zeros_to_append = target_size - len(lst)
        ext = [0.0] * num_zeros_to_append
        lst = torch.cat((lst, torch.tensor(ext, device=lst.device)), dim=0)

    elif len(lst) > target_size:
        lst = lst[:target_size]
    else:
        pass
    return lst.to(dtype=torch.float)

class LargeModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers):
        super(LargeModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.config = None
        layers = []
        prev_size = self.input_size
        for _ in range(self.num_hidden_layers):
            layers.append(nn.Linear(prev_size, self.hidden_size))
            prev_size = self.hidden_size

        layers.append(nn.Linear(hidden_size, 1))  # Output layer

        self.layers = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_list  = input_ids
        logits = []
        for input in input_list:
            input = adjust_list_size(input, self.input_size)
            logits.append(self.layers(input))
        return torch.stack(logits).to(torch.float32)