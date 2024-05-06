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
        return nn.LayerNorm(output_size, elementwise_affine=True)  # Usually, we want affine transformation in LayerNorm
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

class ContinuousFeatureEmbeddingLayer(nn.Module):
    def __init__(self, embedding_size, *num_features, act_fn='layer_norm'):
        super(ContinuousFeatureEmbeddingLayer, self).__init__()
        # Initialize parameters for feature transformation
        self.num_features = num_features
        self.embedding_size = embedding_size

        # Calculate multipliers for each feature type based on maximum number of features
        max_num_features = max(num_features)
        self.multipliers = [max_num_features // nf for nf in num_features]

        # Calculate the total number of features after applying the multipliers
        total_num_features = sum(nf * mult for nf, mult in zip(num_features, self.multipliers))
        
        self.weight = nn.Parameter(torch.randn(total_num_features, embedding_size))
        self.bias = nn.Parameter(torch.zeros(embedding_size))

        # Activation function to normalize or apply non-linearity
        self.final_layer = get_activation_function(act_fn, embedding_size)

    def forward(self, *features):
        # Repeat features according to their multipliers to align their dimensions
        repeated_features = [feature.repeat_interleave(multiplier, dim=-1) for feature, multiplier in zip(features, self.multipliers)]        

        # Concatenate all features along the last dimension after repetition
        aligned_features = torch.cat(repeated_features, dim=-1)

        # Apply weights and add bias
        feature_emb_mul = torch.matmul(aligned_features, self.weight)
        feature_emb_bias = feature_emb_mul + self.bias

        # Apply final layer (e.g., LayerNorm)
        sequence_embeddings = self.final_layer(feature_emb_bias)
        return sequence_embeddings

class ContinuousFeatureJointLayer(nn.Module):
    def __init__(self, embedding_size, *num_features, act_fn='layer_norm', combine_mode='prod'):
        super(ContinuousFeatureJointLayer, self).__init__()
        
        # Initialize embedding layers for each feature set
        self.embedding_layers = nn.ModuleList([
            ContinuousFeatureEmbeddingLayer(embedding_size, num_feature, act_fn='none')
            for num_feature in num_features
        ])
        
        self.final_layer = get_activation_function(act_fn, embedding_size)
        self.combine_mode = combine_mode
        
    def forward(self, *features):
        if len(features) != len(self.embedding_layers):
            raise ValueError("Number of feature inputs must match number of embedding layers")
        
        # Apply embedding layers to corresponding feature inputs
        expanded_features = [embedding_layer(feature) for embedding_layer, feature in zip(self.embedding_layers, features)]
        
        # Combine the transformed features according to the specified mode
        if self.combine_mode == 'sum':
            feature_embeddings = torch.sum(torch.stack(expanded_features), dim=0)
        elif self.combine_mode == 'mean':
            feature_embeddings = torch.mean(torch.stack(expanded_features), dim=0)
        elif self.combine_mode == 'concat':
            feature_embeddings = torch.cat(expanded_features, dim=-1)
        elif self.combine_mode == 'prod':
            feature_embeddings = torch.prod(torch.stack(expanded_features), dim=0)
        else:
            raise ValueError("Unsupported combine mode")

        # Apply the final layer (e.g., LayerNorm) to the combined features
        return self.final_layer(feature_embeddings)  # Shape: [B, S, embedding_size]
