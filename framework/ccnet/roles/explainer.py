'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho@ccnets.org
'''
import torch.nn as nn
from nn.utils.init import init_weights, create_layer
from nn.utils.init import ContinuousFeatureEmbeddingLayer

class Explainer(nn.Module):
    def __init__(self, net, network_params, input_shape, output_size, act_fn = 'layer_norm'):
        super(Explainer, self).__init__()
        d_model = network_params.d_model
        
        self.use_image = len(input_shape) != 1
        
        self.input_embedding_layer = None if self.use_image else ContinuousFeatureEmbeddingLayer(input_shape[-1], d_model)
        
        self.net = net(network_params)
        self.relu = nn.ReLU()
        self.final_layer = create_layer(d_model, output_size, act_fn = act_fn) 
        
        self.apply(init_weights)

    def forward(self, x, padding_mask = None):
        if not self.use_image:
            x = self.input_embedding_layer(x)
        e = self.net(x) if padding_mask is None else self.net(x, padding_mask)
        e = self.relu(e) 
        e = self.final_layer(e)
        return e