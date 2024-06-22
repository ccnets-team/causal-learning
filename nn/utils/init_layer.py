'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
'''

import torch
import random
import numpy as np
from torch import nn
INIT_WEIGHTS_NORMAL_STD = 0.1

def set_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

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
    
def init_weights(module, init_type='xavier_uniform'):
    """
    Applies Xavier uniform or normal initialization to layers in a module that have 'weight' and 'bias' attributes.
    Excludes modules where parameters are directly set if the variable name includes "pretrained".

    Args:
        module (nn.Module): The module to initialize.
        init_type (str): The type of initialization ('xavier_uniform' or 'normal').
    """

    # Initialize weight and bias if present
    if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter) and module.weight.dim() >= 2:
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(module.weight)
        elif init_type == 'normal':
            nn.init.normal_(module.weight, mean=0.0, std=INIT_WEIGHTS_NORMAL_STD)
        else:
            raise ValueError(f"Invalid initialization type: {init_type}")
  
    if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
        nn.init.zeros_(module.bias)

    # Special handling for MultiheadAttention layers
    if isinstance(module, nn.MultiheadAttention):
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(module.in_proj_weight)
        elif init_type == 'normal':
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=INIT_WEIGHTS_NORMAL_STD)
        else:
            raise ValueError(f"Invalid initialization type: {init_type}")
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
            
    # Apply recursively to child submodules regardless of the parent's type
    for child in module.children():
        init_weights(child, init_type = init_type)