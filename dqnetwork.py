import torch.nn as nn


class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)
