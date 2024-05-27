'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
'''

import torch
from torch import nn
from nn.utils.layers import EmbeddingLayer, get_activation_function

try:
    from nn.utils.test import JointLayer
except ImportError:
    class JointLayer(nn.Module):
        def __init__(self, embedding_size, *num_features, act_fn='layer_norm'):
            super(JointLayer, self).__init__()
        
            # Process num_features to take out the last element of sublists
            filtered_num_features = []
            for nf in num_features:
                if isinstance(nf, list):
                    filtered_num_features.append(nf[-1])  # Take all but the last element
                else:
                    filtered_num_features.append(nf)
                                
            self.embedding_layers = nn.ModuleList([EmbeddingLayer(nf, embedding_size) for nf in filtered_num_features])
            self.final_layer = get_activation_function(act_fn, embedding_size)

        def forward(self, *features):
            processed_features_list = [emb_layer(feature) for emb_layer, feature in zip(self.embedding_layers, features)]
            processed_features = torch.prod(torch.stack(processed_features_list), dim=0)
            return self.final_layer(processed_features)
        