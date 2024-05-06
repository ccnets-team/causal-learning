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
    """
    The Producer module in the CCNets architecture, responsible for generating outputs
    based on conditioned and explained inputs. This module can handle both image and non-image
    data, producing outputs accordingly.

    Attributes:
        use_image (bool): Flag indicating whether the output data is image data based on its shape.
        net (nn.Module): The main neural network module that generates output.
        input_embedding_layer (nn.Module): Embedding layer for non-image data.
        relu (nn.Module): A ReLU activation layer for introducing non-linearity.
        final_layer (nn.Module): The final transformation layer to produce the desired output size.
    """

    def __init__(self, net, network_params, condition_size, explain_size, output_shape, act_fn='none'):
        """
        Initializes the Producer module with network architecture and parameters.

        Parameters:
            net (callable): A callable that returns an nn.Module when passed network_params.
            network_params (object): Parameters specific to the neural network being used.
            condition_size (int): The size of the condition part of the input.
            explain_size (int): The size of the explain part of the input.
            output_shape (tuple): The shape of the output data.
            act_fn (str): The activation function name to use in the final layer (default 'none').
        """
        super(Producer, self).__init__()
        d_model = network_params.d_model
        self.use_image = len(output_shape) != 1

        # Embedding layer for combined condition and explanation inputs
        self.input_embedding_layer = ContinuousFeatureEmbeddingLayer(d_model, condition_size, explain_size)

        # Initialize the main network module
        self.net = net(network_params)
        if not self.use_image:
            output_size = output_shape[-1]
            # Activation layer and final layer to adjust to the required output size
            self.relu = nn.ReLU()
            self.final_layer = create_layer(d_model, output_size, act_fn=act_fn)

        # Apply initial weights
        self.apply(init_weights)

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
        
        z = self.input_embedding_layer(labels, explains)
        if self.use_image:
            # Directly process image data through the network
            return self.net(z)
        else:
            # Reverse the tensor sequence for processing
            reversed_z, reversed_padding_mask = self.flip_tensor(z, padding_mask)
            reversed_x = self.net(reversed_z) if reversed_padding_mask is None else self.net(reversed_z, reversed_padding_mask)
            # Reverse the output sequence back to original order
            x, _ = self.flip_tensor(reversed_x)
            x = self.relu(x)
            return self.final_layer(x)

    def flip_tensor(self, z, padding_mask=None):
        """
        Flips the tensor sequence and its corresponding padding mask (if provided) for reverse processing.

        Parameters:
            z (Tensor): The tensor to be reversed.
            padding_mask (Tensor, optional): The padding mask associated with z.

        Returns:
            tuple: The reversed tensor and its corresponding padding mask (if any).
        """
        # Flip the tensor on its sequence dimension (usually time or sequence length)
        return z.flip(dims=[1]), padding_mask.flip(dims=[1]) if padding_mask is not None else None
