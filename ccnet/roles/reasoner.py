'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho@ccnets.org
'''

import torch
import torch.nn as nn
from nn.utils.init_layer import init_weights
from nn.utils.joint_layer import JointLayer
from tools.config.ccnet_config import CooperativeNetworkConfig as NetConfig

class Reasoner(nn.Module):
    """
    The Reasoner module processes and reasons about data, integrating explanations into its predictions,
    for both image and non-image data.

    Attributes:
        use_image (bool): Flag to determine if the input data is image-based.
        obs_shape (tuple): The shape of the input data.
        explain_size (int): The size of the explanation data to be integrated.
        embedding_layer (nn.Module): Embedding layer for integrating observation and explanation data.
        net (nn.Module): The main neural network, adaptable for both image and non-image inputs.
        final_layer (nn.Module): The final transformation layer to produce the desired output.
    """

    def __init__(self, net, config, act_fn='none'):
        """
        Initializes the Reasoner module with network architecture and parameters.

        Parameters:
            net (callable): A callable that returns an nn.Module when passed config.
            config (object): Parameters specific to the neural network being used.
            act_fn (str): The activation function name to use in the final layer (default 'none').
        """
        super(Reasoner, self).__init__()

        self.__model_name = self._get_name()

        input_shape, explain_size, d_model, output_size, device = (config.obs_shape,
                                                                   config.e_dim,
                                                                   config.d_model,
                                                                   config.y_dim,
                                                                   config.device)

        joint_shape = d_model if len(input_shape) == 1 else input_shape
        self.joint_layer = JointLayer(self.__model_name, input_shape, explain_size, output_shape = joint_shape, device = device)
        net_config = NetConfig(config, self.__model_name, self.joint_layer.output_shape, output_size, act_fn)
        self.net = net(net_config)
        
        self.apply(lambda module: init_weights(module))

    def forward(self, obs, e, padding_mask=None):
        """
        Defines the forward pass of the Reasoner module.

        Parameters:
            obs (Tensor): The observation data tensor.
            e (Tensor): The explanation data tensor.
            padding_mask (Tensor, optional): Optional padding mask to be used on input data.

        Returns:
            Tensor: The output tensor after processing through the network.
        """
        z = self.joint_layer(obs, e)
        y = self.net(z, padding_mask=padding_mask)
        return y