'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import torch
import torch.nn as nn
from nn.utils.init_layer import init_weights
from nn.utils.joint_layer import JointLayer
from ccnet.utils import FlipTensor
from tools.config.ccnet_config import CooperativeNetworkConfig as NetConfig

class Producer(nn.Module):
    """
    The Producer module generates outputs based on conditioned and explained inputs.
    This module can handle both image and non-image data, producing outputs accordingly.

    Attributes:
        use_image (bool): Flag indicating whether the output data is image data based on its shape.
        net (nn.Module): The main neural network module that generates output.
        embedding_layer (nn.Module): Embedding layer for combined condition and explanation inputs.
        final_layer (nn.Module): The final transformation layer to produce the desired output size.
    """

    def __init__(self, net, config, act_fn='none'):
        """
        Initializes the Producer module with network architecture and parameters.

        Parameters:
            net (callable): A callable that returns an nn.Module when passed config.
            config (object): Parameters specific to the neural network being used.
            act_fn (str): The activation function name to use in the final layer (default 'none').
        """
        super(Producer, self).__init__()
        
        output_shape, d_model, explain_size, target_size, device = (config.obs_shape, 
                                                                    config.d_model, 
                                                                    config.e_dim, 
                                                                    config.y_dim,
                                                                    config.device)
        self.__model_name = self._get_name()
        
        # Embedding layer for combined condition and explanation inputs
        self.joint_layer = JointLayer(self.__model_name, target_size, explain_size, output_shape = d_model, device = device)
        self.flip_tensor = FlipTensor(output_shape)
        
        producer_config = NetConfig(config, self.__model_name, d_model, output_shape, act_fn)
        
        # Initialize the main network module
        self.net = net(producer_config)
        
        # Apply initial weights
        self.apply(lambda module: init_weights(module))

    def forward(self, labels, explains, padding_mask=None):
        """
        Defines the forward pass of the Producer module.

        Parameters:
            labels (Tensor): The condition data tensor.
            explains (Tensor): The explanation data tensor.
            padding_mask (Tensor, optional): Optional padding mask to be used on input data.

        Returns:
            Tensor: The output tensor after processing through the network.
        """
        z = self.joint_layer(labels, explains)

        reversed_z, reversed_padding_mask = self.flip_tensor(z, padding_mask)
        reversed_x = self.net(reversed_z, padding_mask=reversed_padding_mask)
        x, _ = self.flip_tensor(reversed_x)
        
        return x
    