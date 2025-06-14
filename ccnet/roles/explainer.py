'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''

import torch.nn as nn
from nn.utils.init_layer import init_weights
from tools.config.ccnet_config import CooperativeNetworkConfig as NetConfig
from nn.utils.transform_layer import TransformLayer

class Explainer(nn.Module):
    """
    The Explainer module processes inputs and makes explanations based on the input data,
    whether it's tabular or image data. It adjusts its behavior based on the input type.

    Attributes:
        use_image (bool): Determines if the input data is image data based on the shape of the input.
        input_embedding_layer (nn.Module): Transforms continuous input features into an embedded space.
        final_layer (nn.Module): Final transformation layer to produce the desired output size.
        net (nn.Module): The main neural network module that processes the embedded or raw input.
    """

    def __init__(self, net, config, act_fn='none'):
        """
        Initializes the Explainer module with the specified network architecture and parameters.

        Parameters:
            net (callable): A callable that returns an nn.Module when passed config.
            config (object): Parameters specific to the neural network being used.
            act_fn (str): The activation function name to use in the final layer (default 'none').
        """
        super(Explainer, self).__init__()
        
        self.__model_name = self._get_name()

        input_shape, d_model, output_size = (config.obs_shape, 
                                             config.d_model, 
                                             config.e_dim)

        self.embedding_layer = TransformLayer(input_shape, d_model, last_act_fn='tanh')
        net_config = NetConfig(config, self.__model_name, self.embedding_layer.output_shape, output_size, act_fn)

        self.net = net(net_config)
        
        self.apply(lambda module: init_weights(module, init_type='normal'))

    def forward(self, x, padding_mask=None):
        """
        Defines the forward pass of the Explainer module.

        Parameters:
            x (Tensor): The input data tensor.
            padding_mask (Tensor, optional): Optional padding mask to be used on input data.

        Returns:
            Tensor: The output tensor after processing through the network.
        """
        embeddding = self.embedding_layer(x)
        
        e = self.net(embeddding, padding_mask=padding_mask)

        return e
