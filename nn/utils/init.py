'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
'''

import torch
import random
import numpy as np
from torch import nn
    
def set_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def init_weights(module):
    """
    Applies Xavier uniform initialization to certain layers in a module and its submodules,
    excluding modules where parameters are directly set if the variable name includes "pretrained".
    Args:
        module (nn.Module): The module to initialize.
    """
    if 'pretrained' in module._get_name().lower():
        return

    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        nn.init.xavier_uniform_(module.in_proj_weight)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
            
    # Apply recursively to child submodules regardless of the parent's type
    for child in module.children():
        init_weights(child)