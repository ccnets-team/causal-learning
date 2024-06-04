'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''
import torch
import torch.nn as nn
from nn.utils.init import init_weights
from nn.utils.joint_layer import JointLayer
from nn.utils.final_layer import FinalLayer
from tools.setting.ml_config import modify_network_params

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

    def __init__(self, net, network_params, reset_pretrained, act_fn='none'):
        """
        Initializes the Producer module with network architecture and parameters.

        Parameters:
            net (callable): A callable that returns an nn.Module when passed network_params.
            network_params (object): Parameters specific to the neural network being used.
            act_fn (str): The activation function name to use in the final layer (default 'none').
        """
        super(Producer, self).__init__()
        
        producer_params = modify_network_params(network_params, None)
        
        output_shape, d_model, explain_size, condition_size = (producer_params.obs_shape, 
                                                               producer_params.d_model, 
                                                               producer_params.z_dim, 
                                                               producer_params.condition_dim)

        # Embedding layer for combined condition and explanation inputs
        self.joint_layer = JointLayer(d_model, condition_size, explain_size)

        # Initialize the main network module
        self.net = net(producer_params)

        self.use_image = len(output_shape) != 1
        
        if not self.use_image:
            self.final_layer = FinalLayer(d_model, output_shape, first_act_fn='relu', last_act_fn=act_fn)

        # Apply initial weights
        self.apply(lambda module: init_weights(module, reset_pretrained))

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
        if self.use_image:
            return self.net(z)
        else:
            reversed_z, reversed_padding_mask = self.flip_tensor(z, padding_mask)
            reversed_x = self.net(reversed_z) if reversed_padding_mask is None else self.net(reversed_z, padding_mask=reversed_padding_mask)
            x, _ = self.flip_tensor(reversed_x)
            return self.final_layer(x)
    
    def flip_tensor(self, tensor, padding_mask=None):
        """
        Reverses the order of elements in the tensor along the specified dimension.

        Parameters:
            tensor (Tensor): The tensor to be reversed.
            padding_mask (Tensor, optional): Optional padding mask to reverse.

        Returns:
            Tuple[Tensor, Tensor]: The reversed tensor and the reversed padding mask.
        """
        reversed_tensor = torch.flip(tensor, dims=[1])
        if padding_mask is not None:
            reversed_padding_mask = torch.flip(padding_mask, dims=[1])
            return reversed_tensor, reversed_padding_mask
        return reversed_tensor, None
    