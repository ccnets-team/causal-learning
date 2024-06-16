"""
Residual and Multi-Layer Perceptron Networks

This module implements neural network architectures based on residual learning and
multi-layer perceptron (MLP) principles. It includes a Residual Block (ResBlock),
a full Residual MLP (ResMLP), and a simple MLP structure designed to handle
sequential and structured data effectively.

Components:
- ResBlock: Implements a residual learning block with two linear transformations,
  ReLU activations, and dropout. It facilitates the creation of deep networks
  that can learn efficiently by adding shortcut connections.
- ResMLP: Comprises multiple ResBlocks to form a deep network that leverages
  residual connections to enhance learning in deep layers.
- MLP: A generic multi-layer perceptron that utilizes a sequence of linear layers,
  ReLU activations, and dropout to process inputs. This class is flexible, allowing
  dynamic construction based on layer sizes.

These architectures are well-suited for tasks that require robust feature extraction
and transformation capabilities, commonly used in areas such as regression, classification,
and other predictive tasks.

References:
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
  recognition. Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR).
"""

from torch import nn
from nn.utils.transform_layer import TransformLayer

class ResBlock(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, padding_mask = None):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)  
        return out

class ResMLP(nn.Module):
    def __init__(self, network_config):
        hidden_size, num_layers, dropout = network_config.d_model, network_config.num_layers, network_config.dropout
        super(ResMLP, self).__init__()
        self.layers = nn.Sequential(
            *(ResBlock(hidden_size, dropout=dropout) for _ in range(num_layers))
        )
        self.final_layer = TransformLayer(hidden_size, network_config.output_shape, first_act_fn='relu', last_act_fn=network_config.act_fn)

    def forward(self, x, padding_mask = None):
        out = self.layers(x)
        return self.final_layer(out)
    
class MLP(nn.Module):
    def create_deep_modules(self, layers_size, dropout = 0.0):
        deep_modules = []
        for in_size, out_size in zip(layers_size[:-1], layers_size[1:]):
            deep_modules.append(nn.Linear(in_size, out_size))
            deep_modules.append(nn.ReLU())
            deep_modules.append(nn.Dropout(dropout))
        return nn.Sequential(*deep_modules)

    def __init__(self, network_config):
        hidden_size, num_layer, dropout = network_config.d_model, network_config.num_layers, network_config.dropout
        super(MLP, self).__init__()   
        self.deep = self.create_deep_modules([hidden_size] + [int(hidden_size) for i in range(num_layer)], dropout)
        self.final_layer = TransformLayer(hidden_size, network_config.output_shape, first_act_fn='relu', last_act_fn=network_config.act_fn)
                
    def forward(self, x, padding_mask = None):
        x = self.deep(x)
        return self.final_layer(x)
    