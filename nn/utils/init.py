'''
COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
import os
import torch
import random
import numpy as np
import torch.nn.functional as F
from collections.abc import Iterable
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
    excluding ContinuousFeatureEmbeddingLayer where parameters are directly set.
    Args:
        module (nn.Module): The module to initialize.
    """
    if not isinstance(module, ContinuousFeatureEmbeddingLayer):
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

        
ACTIVATION_FUNCTIONS = {
    "softmax": nn.Softmax(dim=-1),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU()
}

def get_activation_function(activation_function, output_size=None):
    """Returns the appropriate activation function layer."""
    if activation_function.lower() == "none":
        return nn.Identity()
    
    if activation_function == 'layer_norm' and output_size is not None:
        return nn.LayerNorm(output_size, elementwise_affine=False)
    elif activation_function in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[activation_function]
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}")
    
def create_layer(input_size=None, output_size=None, act_fn="none"):
    """Creates a PyTorch layer with optional input and output sizes, and optional activation functions."""
    layers = []
    if input_size is not None and output_size is not None:
        layers.append(nn.Linear(input_size, output_size))
    activation_layer = get_activation_function(act_fn, output_size)
    if isinstance(activation_layer, nn.Module):  # Ensure that a valid layer is returned
        layers.append(activation_layer)
    return nn.Sequential(*layers) if layers else nn.Identity()

class ContinuousFeatureEmbeddingLayer(nn.Module):
    def __init__(self, num_features, embedding_size, act_fn='layer_norm'):
        super(ContinuousFeatureEmbeddingLayer, self).__init__()
        self.feature_embeddings = nn.Parameter(torch.randn(num_features, embedding_size))
        self.bias = nn.Parameter(torch.zeros(1, embedding_size))
        self.final_layer = get_activation_function(act_fn, embedding_size)
                    
    def forward(self, features):
        features_expanded = features.unsqueeze(-1)
        feature_emb_mul = features_expanded * self.feature_embeddings   
        feature_emb_bias = feature_emb_mul.sum(dim=-2) + self.bias
        sequence_embeddings = self.final_layer(feature_emb_bias)
        return sequence_embeddings
    
class ContinuousFeatureJointLayer(nn.Module):
    def __init__(self, num_features1, num_features2, embedding_size, act_fn='relu'):
        super(ContinuousFeatureJointLayer, self).__init__()
        max_features = max(num_features1, num_features2)
        self.expansion_layer1 = create_layer(num_features1, max_features, act_fn)
        self.expansion_layer2 = create_layer(num_features2, max_features, act_fn)
        
        self.embedding_layer = ContinuousFeatureEmbeddingLayer(2*max_features, embedding_size)

    def forward(self, feature1, feature2):
        expanded_feature1 = self.expansion_layer1(feature1)
        expanded_feature2 = self.expansion_layer2(feature2)

        # Concatenate the features
        features_combined = torch.cat([expanded_feature1, expanded_feature2], dim=-1)

        # Pass the concatenated features through the embedding layer
        sequence_embeddings = self.embedding_layer(features_combined)

        return sequence_embeddings