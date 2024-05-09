"""
SuperNet Implementation

This module implements the SuperNet architecture, which consists of a series of blocks
that can perform different operations dynamically during training. Each block in the
network has the capability to adjust its internal pathways according to learned architectural
weights, enabling a form of learned network topology adaptation.

Components:
- SuperNetBlock: A single block of SuperNet that can dynamically combine different operations
  based on learnable weights. This enables adaptive computational pathways within the network.
- SuperNet: Comprises multiple SuperNetBlocks to form a complete model capable of handling
  sequential data with variable internal structures.

The architecture allows for flexibility and adaptability in processing diverse inputs, making
it suitable for tasks that benefit from dynamic network topology adjustments.

References:
- Inspired by concepts from neural architecture search and dynamic network configurations
  found in various deep learning research.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperNetBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(SuperNetBlock, self).__init__()
        # Different possible operations within the block
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Architecture weights for each operation
        self.arch_weights = nn.Parameter(torch.ones(2) / 2)  # Initialize close to a uniform distribution
        
    def forward(self, x):
        weights = F.softmax(self.arch_weights, dim=-1)
        out = weights[0] * self.linear1(x) + weights[1] * self.linear2(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class SuperNet(nn.Module):
    def __init__(self, network_params):
        num_layers, d_model, dropout = network_params.num_layers, network_params.d_model, network_params.dropout
        super(SuperNet, self).__init__()
        self.num_layers = num_layers

        layers = []
        for i in range(self.num_layers):
            layers.append(SuperNetBlock(d_model, d_model, dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x, padding_mask = None):
        return self.net(x)