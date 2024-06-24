'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
'''

import torch
from torch import nn
from nn.utils.transform_layer import get_activation_function
from ccnet.utils import convert_shape_to_size, find_image_indices, FeatureToImageShape

try:
    from nn.utils.test import JointLayer
except ImportError:
    class JointLayer(nn.Module):
        def __init__(self, parent_name, *input_shapes, output_shape = None, act_fn='tanh', device = 'cuda'):
            super(JointLayer, self).__init__()
            
            input_sizes = convert_shape_to_size(input_shapes)
            output_size = convert_shape_to_size(output_shape)
            image_indices = find_image_indices(input_shapes)
            self.use_image = len(image_indices) > 0
            self.parent_name = parent_name
            
            if self.use_image:
                assert len(image_indices) == 1, f"Only one image input is allowed for {self.parent_name}."
                num_features = len(input_sizes) - len(image_indices)
                
                image_idx = image_indices[0]
                image_shape = input_shapes[image_idx]
                embedding_layers = nn.ModuleList([
                    FeatureToImageShape(input_sizes[idx], image_shape)
                    if idx not in image_indices else nn.Identity()
                    for idx in range(len(input_sizes))
                ])
                final_layer = nn.Identity()
                self.output_shape = [image_shape[0] + num_features] + list(image_shape[1:])
            else:
                embedding_layers = nn.ModuleList([nn.Sequential(
                        nn.Linear(input_size, output_size)
                    ) for input_size in input_sizes])
                final_layer = get_activation_function(act_fn, output_size)
                self.output_shape = output_size
                
            self.embedding_layers = embedding_layers.to(device)
            self.final_layer = final_layer.to(device)
        
        def forward(self, *inputs):
            embedded_inputs = [emb_layer(feature) for emb_layer, feature in zip(self.embedding_layers, inputs)]
            if self.use_image:
                joint_feature = torch.cat(embedded_inputs, dim=-3)
            else:
                joint_feature = torch.prod(torch.stack(embedded_inputs), dim=0)
            return self.final_layer(joint_feature)