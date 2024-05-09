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

class DeepFM(nn.Module):
    def __init__(self, network_params):
        super(DeepFM, self).__init__()
        hidden_size, num_layers = network_params.d_model, network_params.num_layers 

        state_dim = hidden_size
        output_dim = hidden_size
        self.state_dim = state_dim
        self.num_layers = num_layers
        
        # Define hidden layers for 2nd order terms
        self.dnn = nn.ModuleList()
        for i in range(num_layers):
            self.dnn.append(nn.Linear(hidden_size, hidden_size))
        # Define output layer
        self.output_layer = nn.Linear(hidden_size, output_dim)
        
        
    def forward(self, x):
        # Expand the input tensor along the feature dimension
        # Interaction terms
        
        interactions = torch.matmul(x.unsqueeze(2), x.unsqueeze(1))
        interactions = F.relu(interactions.sum(2))
        # Hidden layers for 2nd order terms
        hidden_output = x.view(-1, self.state_dim)
        for i in range(self.num_layers):
            hidden_output = F.relu(self.dnn[i](hidden_output))
        
        # Output layer
        output = self.output_layer((hidden_output + interactions)/2.)
        
        # Add linear terms and hidden layer outputs element-wise
        return output