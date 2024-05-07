'''
COPYRIGHT (c) 2022. CCNets. All Rights reserved.
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
    excluding ContinuousFeatureEmbeddingLayer where parameters are directly set.
    Args:
        module (nn.Module): The module to initialize.
    """
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
    activation_function = activation_function.lower()
    if activation_function in ["none", "linear"]:  # Treating 'linear' as no activation
        return nn.Identity()
    if activation_function == 'layer_norm' and output_size is not None:
        return nn.LayerNorm(output_size, elementwise_affine=False)  # Usually, we want affine transformation in LayerNorm
    elif activation_function in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[activation_function]
    else:
        raise ValueError(f"Unsupported activation function: {activation_function}")

def create_layer(input_size, output_size, act_fn="none"):
    """Creates a PyTorch layer with specified input and output sizes, including an optional activation function."""
    layers = [nn.Linear(input_size, output_size)]  # Always include the linear transformation
    activation_layer = get_activation_function(act_fn, output_size)
    if not isinstance(activation_layer, nn.Identity):  # Add activation layer if it's not Identity
        layers.append(activation_layer)
    return nn.Sequential(*layers)

class MatMulLayer(nn.Module):
    """ Layer that applies a linear transformation to the input features """
    def __init__(self, input_features, output_features):
        super(MatMulLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(input_features, output_features))
        self.bias = nn.Parameter(torch.zeros(output_features))

    def forward(self, feature):
        # Apply weights and add bias
        feature_emb_mul = torch.matmul(feature, self.weight)
        feature_emb_bias = feature_emb_mul + self.bias
        return feature_emb_bias

class ConCatLayer(nn.Module):
    """ Layer that concatenates features with optional repeating to align their dimensions """
    def __init__(self, embedding_size, *num_features, use_repeat=False):
        super(ConCatLayer, self).__init__()
        self.num_features = num_features

        # Calculate multipliers for each feature type based on the maximum number of features
        max_num_features = max(num_features)
        self.multipliers = [max_num_features // nf for nf in num_features] if use_repeat else [1 for _ in num_features]

        # Calculate the total number of features after applying the multipliers
        num_tranformed_features = sum(nf * mult for nf, mult in zip(num_features, self.multipliers))
        
        self.mat_mul_layer = MatMulLayer(num_tranformed_features, embedding_size)

    def forward(self, *features):
        # Optionally repeat features according to their multipliers to align their dimensions
        repeated_features = [feature.repeat_interleave(multiplier, dim=-1) if multiplier > 1 else feature 
                             for feature, multiplier in zip(features, self.multipliers)]
        # Concatenate all features along the last dimension after repetition
        aligned_features = torch.cat(repeated_features, dim=-1)
        
        aligned_features = self.mat_mul_layer(aligned_features)
        
        return aligned_features
    
class ContinuousFeatureJointLayer(nn.Module):
    def __init__(self, embedding_size, *num_features, act_fn='layer_norm', combine_mode='cat'):
        super(ContinuousFeatureJointLayer, self).__init__()
        self.combine_mode = combine_mode
        if self.combine_mode == 'cat':
            self.concat_layer = ConCatLayer(embedding_size, *num_features, use_repeat=False)
        else:
            self.mat_mul_layers = nn.ModuleList([MatMulLayer(nf, embedding_size) for nf in num_features])
        self.final_layer = get_activation_function(act_fn, embedding_size)

    def forward(self, *features):
        if self.combine_mode == 'cat':
            processed_features = self.concat_layer(*features)
        else:
            processed_features_list = [mat_mul_layer(feature) for mat_mul_layer, feature in zip(self.mat_mul_layers, features)]
            if self.combine_mode == 'sum':
                processed_features = torch.sum(torch.stack(processed_features_list), dim=0)
            elif self.combine_mode == 'mean':
                processed_features = torch.mean(torch.stack(processed_features_list), dim=0)
            elif self.combine_mode == 'prod':
                processed_features = torch.prod(torch.stack(processed_features_list), dim=0)
            else:
                raise ValueError("Unsupported combine mode")

        return self.final_layer(processed_features)