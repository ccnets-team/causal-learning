'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho@ccnets.org
'''
import torch
import torch.nn as nn
from nn.utils.init import init_weights, create_layer
from nn.utils.init import ContinuousFeatureEmbeddingLayer

class Reasoner(nn.Module):
    def __init__(self, net, network_params, input_shape, explain_size, output_size, act_fn = 'sigmoid'):
        super(Reasoner, self).__init__()
        d_model = network_params.d_model
        self.input_shape = input_shape
        self.explain_size = explain_size
        self.use_image = len(input_shape) != 1

        self.input_embedding_layer = None
        if self.use_image:
            self.image_elements = torch.prod(torch.tensor(input_shape[1:], dtype=torch.int)).item()
            self.net = net(network_params) # ImageNet
        else:
            z_size = input_shape[-1] + explain_size
            self.input_embedding_layer = ContinuousFeatureEmbeddingLayer(z_size, d_model)
            self.net = net(network_params)
            self.relu = nn.ReLU()
            self.final_layer = create_layer(d_model, output_size, act_fn = act_fn) 

        self.apply(init_weights)
    
    def forward(self, obs, e, padding_mask = None):
        if self.use_image:
            y = self.net(obs, e)
        else:
            x = torch.cat([obs, e], dim=-1)
            x = self.input_embedding_layer(x)
            y = self.net(x) if padding_mask is None else self.net(x, padding_mask)
            y = self.relu(y) 
            y = self.final_layer(y)
        return y