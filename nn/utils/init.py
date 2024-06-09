'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
'''

import torch
import random
import numpy as np
from torch import nn
from nn.utils.final_layer import EmbeddingLayer
    
def set_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def init_weights(module, reset_pretrained = False, init_type = 'xavier_uniform'):
    """
    Applies Xavier uniform initialization to certain layers in a module and its submodules,
    excluding modules where parameters are directly set if the variable name includes "pretrained".
    Args:
        module (nn.Module): The module to initialize.
        reset_pretrained (bool): Whether to reset initialization for modules with "pretrained" in their names.
    """
    if reset_pretrained and 'pretrained' in module._get_name().lower():
        return

    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(module.weight)
        elif init_type == 'normal':
            nn.init.normal_(module.weight, mean=0.0, std=1.0)
        else:
            raise ValueError(f"Invalid initialization type: {init_type}")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(module.in_proj_weight)
        elif init_type == 'normal':
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=1.0)
        else:
            raise ValueError(f"Invalid initialization type: {init_type}")
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
    elif isinstance(module, EmbeddingLayer):
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(module.weight)
        elif init_type == 'normal':
            nn.init.normal_(module.weight, mean=0.0, std=1.0)
        else:
            raise ValueError(f"Invalid initialization type: {init_type}")
        nn.init.zeros_(module.bias)
            
    # Apply recursively to child submodules regardless of the parent's type
    for child in module.children():
        init_weights(child, reset_pretrained = reset_pretrained, init_type = init_type)