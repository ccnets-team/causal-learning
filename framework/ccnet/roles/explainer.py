'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''

import torch.nn as nn
from nn.utils.init import init_weights
from tools.setting.ml_config import modify_network_params
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

    def __init__(self, net, network_params, act_fn='none'):
        """
        Initializes the Explainer module with the specified network architecture and parameters.

        Parameters:
            net (callable): A callable that returns an nn.Module when passed network_params.
            network_params (object): Parameters specific to the neural network being used.
            act_fn (str): The activation function name to use in the final layer (default 'none').
        """
        super(Explainer, self).__init__()
        
        self.__model_name = self._get_name()
        explainer_params = modify_network_params(network_params, None)
        input_shape, d_model, output_size, reset_pretrained = (explainer_params.obs_shape, 
                                                               explainer_params.d_model, 
                                                               explainer_params.z_dim,
                                                               explainer_params.reset_pretrained)

        self.embedding_layer = TransformLayer(self.__model_name, input_shape, d_model, last_act_fn='tanh')

        self.net = net(explainer_params)
                
        self.final_layer = TransformLayer(self.__model_name, d_model, output_size, first_act_fn='relu', last_act_fn=act_fn)
        
        self.apply(lambda module: init_weights(module, reset_pretrained, init_type='normal'))

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

        e = self.final_layer(e)
        return e
