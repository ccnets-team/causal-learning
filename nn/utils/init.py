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

# Mapping activation functions to their PyTorch counterparts
ACTIVATION_FUNCTIONS = {
    "softmax": nn.Softmax(dim=-1),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU()
}

def set_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def add_activation_to_layers(layers, activation_function, layer_norm_size = None):
    """Appends the specified activation function to the given layers."""
    if activation_function.lower() != "none":
        if activation_function in ACTIVATION_FUNCTIONS:
            layers.append(ACTIVATION_FUNCTIONS[activation_function])
        elif activation_function == 'layer_norm' and layer_norm_size is not None:
            layers.append(nn.LayerNorm(layer_norm_size, elementwise_affine=False))
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")
    
def create_layer(input_size = None, output_size = None, act_fn ="none"):
    """Creates a PyTorch layer with optional input and output sizes, and optional activation functions."""
    layers = []
    # add_activation_to_layers(layers, first_act)
    if (input_size is not None) and (output_size is not None):  
        layers.append(nn.Linear(input_size, output_size))
    add_activation_to_layers(layers, act_fn, output_size)
    return nn.Sequential(*layers)

def init_weights(param_object):
    if isinstance(param_object, Iterable):
        for layer in param_object:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.MultiheadAttention):
                nn.init.xavier_uniform_(layer.in_proj_weight)
                nn.init.zeros_(layer.in_proj_bias)
            elif isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            else:
                # Handle other layer types if needed
                pass
    else:
        if isinstance(param_object, nn.Linear):
            nn.init.xavier_uniform_(param_object.weight)
            nn.init.zeros_(param_object.bias)
        elif isinstance(param_object, nn.MultiheadAttention):
            nn.init.xavier_uniform_(param_object.in_proj_weight)
            nn.init.zeros_(param_object.in_proj_bias)
        elif isinstance(param_object, nn.Conv2d):
            nn.init.xavier_uniform_(param_object.weight)
            nn.init.zeros_(param_object.bias)
        elif isinstance(param_object, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(param_object.weight)
            nn.init.zeros_(param_object.bias)
        else:
            # Handle other layer types if needed
            pass

def setup_directories(base_path = './'):
    set_model_path = os.path.join(base_path, "models")
    set_temp_path = os.path.join(base_path, "models/temp")
    set_log_path = os.path.join(base_path, "logs")

    for path in [set_model_path, set_temp_path, set_log_path]:
        os.makedirs(path, exist_ok=True)

    return set_model_path, set_temp_path, set_log_path

class ContinuousFeatureEmbeddingLayer(nn.Module):
    def __init__(self, num_features, embedding_size, act_fn='tanh'):
        super(ContinuousFeatureEmbeddingLayer, self).__init__()
        self.feature_embeddings = nn.Parameter(torch.randn(num_features, embedding_size))
        self.bias = nn.Parameter(torch.zeros(1, embedding_size))  # Shared bias across features
        self.act_fn = act_fn
    def forward(self, features):
        # Input features shape: [B, S, F]
        # B: Batch size, S: Sequence length, F: Number of features
        features_expanded = features.unsqueeze(-1)
        # After unsqueeze, features shape: [B, S, F, 1]
        # self.feature_embeddings shape: [F, embedding_size]
        # We broadcast multiply features with embeddings to get a shape: [B, S, F, embedding_size]
        feature_emb_mul = features_expanded * self.feature_embeddings
        # Sum across the feature dimension F, resulting shape: [B, S, embedding_size]
        feature_emb_bias = feature_emb_mul.sum(dim=-2) + self.bias  # Sum first, then add bias
        if self.act_fn == "tanh":
            sequence_embeddings = torch.tanh(feature_emb_bias)
        elif self.act_fn == "relu":
            sequence_embeddings = torch.relu(feature_emb_bias)
        else:
            sequence_embeddings = feature_emb_bias
        return sequence_embeddings