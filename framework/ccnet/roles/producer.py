'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''

import torch
import torch.nn as nn
from nn.utils.init import init_weights, create_layer
from nn.utils.init import ContinuousFeatureEmbeddingLayer

class Producer(nn.Module):
    def __init__(self, net, network_params, condition_size, explain_size, output_shape, act_fn = 'none'):
        super(Producer, self).__init__()
        d_model = network_params.d_model
        self.use_image_decoding = len(output_shape) != 1

        self.net = net(network_params)
        if not self.use_image_decoding:
            input_size = condition_size + explain_size
            output_size = output_shape[-1]
            self.input_embedding_layer = ContinuousFeatureEmbeddingLayer(input_size, d_model)
            self.relu = nn.ReLU()
            self.final_layer = create_layer(d_model, output_size, act_fn = act_fn)
        self.apply(init_weights)

    def forward(self, labels, explains, padding_mask=None):
        if self.use_image_decoding:
            return self.net(labels, explains)
        else:
            z = torch.cat([labels, explains], dim=-1)
            z = self.input_embedding_layer(z)
            reversed_z, reversed_padding_mask = self.flip_tensor(z, padding_mask)
            reversed_x = self.net(reversed_z) if reversed_padding_mask is None else self.net(reversed_z, reversed_padding_mask)
            x, _ = self.flip_tensor(reversed_x)
            x = self.relu(x)
            return self.final_layer(x)
             

    def flip_tensor(self, z, padding_mask = None):
        """
        Embeds the input features and reverses the embedded sequence.
        """
        return z.flip(dims=[1]), padding_mask.flip(dims=[1]) if padding_mask is not None else None

