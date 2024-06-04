'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
'''

import torch
from torch import nn

ACTIVATION_FUNCTIONS = {
    "softmax": nn.Softmax(dim=-1),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU()
}

def get_activation_function(activation_function, feature_size=None):
    """Returns the appropriate activation function layer."""
    activation_function = activation_function.lower()
    if activation_function in ["none", "linear"]:  # Treating 'linear' as no activation
        return nn.Identity()
    if activation_function == 'layer_norm' and feature_size is not None:
        return nn.LayerNorm(feature_size, elementwise_affine=False)  # Usually, we want affine transformation in LayerNorm
    elif activation_function in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[activation_function]
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}")

class EmbeddingLayer(nn.Module):
    """ Layer that applies a linear transformation to the input features """
    def __init__(self, input_features, output_features):
        super(EmbeddingLayer, self).__init__()
        # weight shape: [F, W]
        self.weight = nn.Parameter(torch.randn(input_features, output_features))
        # bias shape: [W]
        self.bias = nn.Parameter(torch.zeros(output_features))

    def forward(self, features):
        # features shape: [B, S, F] or [B, F]

        # Add an extra dimension at the end
        features_expanded = features.unsqueeze(-1)
        # features_expanded shape: [B, S, F, 1] or [B, F, 1]

        # Element-wise multiplication with the weight matrix
        feature_emb_mul = features_expanded * self.weight
        # feature_emb_mul shape: [B, S, F, W] or [B, F, W]

        # Sum along the second-to-last dimension to reduce dimensionality
        feature_emb_bias = feature_emb_mul.sum(dim=-2) + self.bias
        # feature_emb_bias shape: [B, S, W] or [B, W]

        return feature_emb_bias
    
class FinalLayer(nn.Module):
    """
    Final layer that configures an input layer with optional pre- and post-activation functions
    and an embedding layer in between.
    """
    def __init__(self, input_size, output_size, first_act_fn="none", last_act_fn="none"):
        super(FinalLayer, self).__init__()
        _input_size = input_size[-1] if isinstance(input_size, list) else input_size
        _output_size = output_size[-1] if isinstance(output_size, list) else output_size
        
        layers = []
        first_activation_layer = get_activation_function(first_act_fn, _input_size)
        if not isinstance(first_activation_layer, nn.Identity):  # Add activation layer if it's not Identity
            layers.append(first_activation_layer)        
        layers.append(EmbeddingLayer(_input_size, _output_size))
        last_activation_layer = get_activation_function(last_act_fn, _output_size)
        if not isinstance(last_activation_layer, nn.Identity):  # Add activation layer if it's not Identity
            layers.append(last_activation_layer)
        self.layers = nn.Sequential(*layers)
            
    def forward(self, features):
        return self.layers(features)
    