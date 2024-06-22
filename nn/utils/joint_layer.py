'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
'''

import torch
from torch import nn
from nn.utils.transform_layer import get_activation_function
from ccnet.utils import convert_shape_to_size

try:
    from nn.utils.test import JointLayer
except ImportError:
    class JointLayer(nn.Module):
        def __init__(self, parent_name, output_shape, *input_shapes, act_fn='tanh'):
            super(JointLayer, self).__init__()
            input_sizes = convert_shape_to_size(input_shapes)
            output_size = convert_shape_to_size(output_shape) 
                                
            self.embedding_layers = nn.ModuleList([nn.Sequential(
                    nn.Linear(input_size, output_size)
                ) for input_size in input_sizes])
            
            self.final_layer = get_activation_function(act_fn, output_size)
            self.parent_name = parent_name
            self.output_size = output_size

        def forward(self, *features):
            embeded_features = [emb_layer(feature) for emb_layer, feature in zip(self.embedding_layers, features)]
            prod_feature = torch.prod(torch.stack(embeded_features), dim=0)
            return self.final_layer(prod_feature)
        