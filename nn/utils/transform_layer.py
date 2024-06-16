'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
'''

import torch
from torch import nn
from nn.utils.init import get_activation_function

class ScaleTransformLayer(nn.Module):
    def __init__(self, feature_size):
        super(ScaleTransformLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones(feature_size))
        self.bias = nn.Parameter(torch.zeros(feature_size))

    def forward(self, x):
        tranformed_x = x * self.weight + self.bias
        return tranformed_x
    
class FeatureTransformLayer(nn.Module):
    """ Layer that applies a linear transformation to the input features """
    def __init__(self, input_features, output_features):
        super(FeatureTransformLayer, self).__init__()
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
    
class TransformLayer(nn.Module):
    """
    Final layer that configures an input layer with optional pre- and post-activation functions
    and an embedding layer in between.
    """
    def __init__(self, input_shape, output_shape, first_act_fn="none", last_act_fn="none"):
        super(TransformLayer, self).__init__()
        # Determine if the input or output shapes suggest the use of image processing
        use_image = (
            (isinstance(input_shape, (list, torch.Size, tuple)) and len(input_shape) != 1) or
            (isinstance(output_shape, (list, torch.Size, tuple)) and len(output_shape) != 1)
        )
        # Build layers
        layers = []
        if not use_image:
            # Extract sizes from shapes if not using images
            input_size = input_shape[-1] if isinstance(input_shape, (list, torch.Size, tuple)) else input_shape
            output_size = output_shape[-1] if isinstance(output_shape, (list, torch.Size, tuple)) else output_shape

            # Add the first activation layer if it's not Identity
            first_activation_layer = get_activation_function(first_act_fn, input_size)
            if not isinstance(first_activation_layer, nn.Identity):
                layers.append(first_activation_layer)

            # Add transformation and scale layers
            layers.append(FeatureTransformLayer(input_size, output_size))
            if last_act_fn == "none":
                layers.append(ScaleTransformLayer(output_size))

            # Add the last activation layer if it's not Identity
            last_activation_layer = get_activation_function(last_act_fn, output_size)
            if not isinstance(last_activation_layer, nn.Identity):
                layers.append(last_activation_layer)
            self.output_shape = output_size
        else:
            self.output_shape = input_shape
        self.input_shape = input_shape
        self.layers = nn.Sequential(*layers)
        
    def forward(self, features):
        return self.layers(features)