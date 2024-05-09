"""
Gated Residual Networks and Variable Selection Flow

This module implements several key components used in deep learning architectures
for feature selection and information gating. It includes custom implementations of Gated
Linear Units (GLUs), Gated Residual Networks (GRNs), and a Variable Selection
mechanism that orchestrates feature selection across multiple inputs.

Components:
- Gated Linear Unit (GLU): A type of activation function with a gating mechanism,
  inspired by 'Language Modeling with Gated Convolutional Networks' (Dauphin et al., ICML 2017).
- Gated Residual Network (GRN): Utilizes gates in residual connections, primarily used
  in 'TabNet: Attentive Interpretable Tabular Learning' (Arik and Pfister, arXiv 2019).
- Variable Selection Network: Orchestrates the selection and processing of features from
  multiple inputs, inspired by mechanisms used in models for structured data.

Modifications and enhancements have been made to the original designs to fit the specific needs
of this project.

References:
- Dauphin, Yann N., et al. "Language modeling with gated convolutional networks." Proceedings
  of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.
- Arik, Sercan O., and Tomas Pfister. "TabNet: Attentive Interpretable Tabular Learning."
  arXiv preprint arXiv:1908.07442, 2020.

"""

import torch
from torch import nn
import torch.nn.functional as F

class GatedLinearUnit(nn.Module):
    def __init__(self, units):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(units, units)
        self.sigmoid = nn.Linear(units, units)
        
    def forward(self, inputs):
        return self.linear(inputs) * torch.sigmoid(self.sigmoid(inputs))
    
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_units, units, dropout):
        super(GatedResidualNetwork, self).__init__()
        self.relu_dense = nn.Linear(input_units, units)
        self.linear_dense = nn.Linear(units, units)
        self.dropout = nn.Dropout(dropout)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = nn.LayerNorm(units)
        self.project = nn.Linear(input_units, units)
        
    def forward(self, inputs):
        x = F.relu(self.relu_dense(inputs))
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.size(-1) != self.gated_linear_unit.linear.out_features:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x
    
class VariableSelection(nn.Module):
    def __init__(self, num_features, units, dropout):
        super(VariableSelection, self).__init__()
        self.grns = nn.ModuleList([GatedResidualNetwork(1, units, dropout) for _ in range(num_features)])
        self.grn_concat = GatedResidualNetwork(num_features, units, dropout)
        self.softmax = nn.Linear(units, num_features)
        self.num_features = num_features
        
    def forward(self, inputs):
        v = torch.cat(inputs, dim=-1)
        v = self.grn_concat(v)
        v = torch.sigmoid(self.softmax(v)).unsqueeze(-1)

        x = []
        for idx, input_ in enumerate(inputs):
            x.append(self.grns[idx](input_))
        x = torch.stack(x, dim=1)

        outputs = (v* x).squeeze(-1)
        return outputs

class VariableSelectionFlow(nn.Module):
    def __init__(self, network_params, units = 1, dense_units=None):
        super(VariableSelectionFlow, self).__init__()
        hidden_size, dropout = network_params.d_model, network_params.dropout
        self.variableselection = VariableSelection(hidden_size, units, dropout)
        self.dense_units = dense_units
        if dense_units:
            self.dense_list = nn.ModuleList([nn.Linear(units, dense_units) for _ in range(hidden_size)])
        self.num_features = hidden_size
        
    def forward(self, inputs, padding_mask = None):
        split_input = torch.split(inputs, 1, dim=-1)
        if self.dense_units:
            l = [self.dense_list[i](split_input[i]) for i in range(self.num_features)]
        else:
            l = list(split_input)
        return self.variableselection(l)
