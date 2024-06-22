'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import torch
import torch.nn as nn
from nn.utils.init_layer import init_weights
from nn.utils.joint_layer import JointLayer
from tools.setting.ccnet_config import CooperativeNetworkConfig

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

    def __init__(self, net, network_params, act_fn='none'):
        """
        Initializes the Producer module with network architecture and parameters.

        Parameters:
            net (callable): A callable that returns an nn.Module when passed network_params.
            network_params (object): Parameters specific to the neural network being used.
            act_fn (str): The activation function name to use in the final layer (default 'none').
        """
        super(Producer, self).__init__()
        
        output_shape, d_model, explain_size, target_size = (network_params.obs_shape, 
                                                            network_params.d_model, 
                                                            network_params.e_dim, 
                                                            network_params.y_dim)
        self.__model_name = self._get_name()
        
        # Embedding layer for combined condition and explanation inputs
        self.embedding_layer = JointLayer(self.__model_name, d_model, target_size, explain_size)
        
        producer_config = CooperativeNetworkConfig(network_params, self.__model_name, d_model, output_shape, act_fn)
        
        # Initialize the main network module
        self.net = net(producer_config)

        self.use_image = len(output_shape) != 1
        
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
        z = self.embedding_layer(labels, explains)

        reversed_z, reversed_padding_mask = self.flip_tensor(z, padding_mask)
        reversed_x = self.net(reversed_z, padding_mask=reversed_padding_mask)
        x, _ = self.flip_tensor(reversed_x)
        
        return x
    
    def flip_tensor(self, tensor, padding_mask=None):
        """
        Reverses the order of elements in the tensor along the specified dimension.

        Parameters:
            tensor (Tensor): The tensor to be reversed.
            padding_mask (Tensor, optional): Optional padding mask to reverse.

        Returns:
            Tuple[Tensor, Tensor]: The reversed tensor and the reversed padding mask.
        """
        if self.use_image:
            return tensor, padding_mask
        
        reversed_tensor = torch.flip(tensor, dims=[1])
        if padding_mask is not None:
            reversed_padding_mask = torch.flip(padding_mask, dims=[1])
            return reversed_tensor, reversed_padding_mask
        return reversed_tensor, None
    