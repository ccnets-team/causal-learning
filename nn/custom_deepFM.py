"""
DeepFM (Deep Factorization Machine) Implementation

This module implements the DeepFM architecture, which is a neural network model that
combines the power of factorization machines for recommendation systems with the flexibility
and learning capacity of deep neural networks. The primary purpose of this implementation
is to handle both high- and low-order feature interactions effectively within the same model.

Features:
- Factorization Machine Component: Handles second-order feature interactions.
- Deep Neural Network: Manages high-order interactions and nonlinear feature transformations.

The DeepFM model is widely used in various tasks such as CTR (Click Through Rate) prediction
and ranking tasks in recommendation systems, where capturing complex interactions between
features is crucial.

References:
- "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction," Huifeng Guo,
  Ruiming Tang, Yunming Ye, Zhenguo Li, Xiuqiang He, IJCAI 2017.

This implementation may contain modifications to the original design to suit specific
needs or to optimize performance for particular tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    
class ContinuousDeepFM(nn.Module):
    def __init__(self, network_params):
        super(ContinuousDeepFM, self).__init__()
        d_model, num_layers, dropout = network_params.d_model, network_params.num_layers, network_params.dropout
        self.num_features = d_model
        self.embed_dim = int(math.sqrt(d_model)) 

        # Initialize 2D parameters for continuous features
        self.first_order_weights = nn.Parameter(torch.randn(d_model, d_model))
        self.bias = nn.Parameter(torch.zeros((d_model,)))

        # Second-order weights
        self.second_order_weights = nn.Parameter(torch.randn(d_model, d_model))  # Adjusted

        # Deep network part
        self.feature_weights = nn.Parameter(torch.randn(d_model, d_model))
        layers = []
        # Define reasonable layer sizes
        for i in range(num_layers):
            layers.append(nn.Linear(d_model, d_model))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(d_model, d_model))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # First-order term computations
        first_order = torch.matmul(x, self.first_order_weights) + self.bias

        # Second-order term computations
        interactions = x.unsqueeze(-1) * x.unsqueeze(-2)  # Creating pairwise feature interactions
        interactions = torch.matmul(interactions, self.second_order_weights)
        sum_squared = torch.sum(interactions ** 2, dim=-2)
        squared_sum = torch.sum(interactions, dim=-2) ** 2
        second_order = 0.5 * (sum_squared - squared_sum)

        # Deep component
        feature_interactions = torch.matmul(x, self.feature_weights)
        deep_component = self.mlp(feature_interactions)

        # Output layer: Combine all three components
        result = first_order + second_order + deep_component
        return result