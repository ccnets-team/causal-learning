'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho@ccnets.org
'''

import torch
import torch.nn as nn
from nn.utils.init import init_weights, create_layer
from nn.utils.init import ContinuousFeatureJointLayer
from copy import deepcopy

class Reasoner(nn.Module):
    """
    The Reasoner module is part of CCNet's architecture, designed to process and reason about data,
    integrating explanations into its predictions, for both image and non-image data.

    Attributes:
        input_shape (tuple): The shape of the input data.
        explain_size (int): The size of the explanation data to be integrated.
        use_image (bool): Flag to determine if the input data is image-based.
        input_embedding_layer (nn.Module): Embedding layer for non-image data.
        image_elements (int): Total elements in an image if using image data.
        net (nn.Module): The main neural network, adaptable for both image and non-image inputs.
        relu (nn.Module): Activation layer for adding non-linearity.
        final_layer (nn.Module): The final transformation layer to produce the desired output.
    """

    def __init__(self, net, network_params, input_shape, explain_size, output_size, act_fn='sigmoid'):
        """
        Initializes the Reasoner module with network architecture and parameters.

        Parameters:
            net (callable): A callable that returns an nn.Module when passed network_params.
            network_params (object): Parameters specific to the neural network being used.
            input_shape (tuple): The shape of the input data.
            explain_size (int): Size of the explanation component.
            output_size (int): The size of the output tensor after final transformation.
            act_fn (str): The activation function name to use in the final layer (default 'sigmoid').
        """
        super(Reasoner, self).__init__()
        d_model = network_params.d_model
        self.obs_shape = network_params.obs_shape
        self.use_image = len(input_shape) != 1
        self.explain_size = explain_size

        if self.use_image:
            # Create a deep copy of network_params to prevent modifications from affecting the original
            _network_params = deepcopy(network_params)
            
            # Increase the number of channels by 1. Ensure this is done correctly:
            _network_params.obs_shape = list(_network_params.obs_shape)  # Convert to list if it's a tuple
            _network_params.obs_shape[0] += 1  # Increment the channel count
            
            # Initialize the neural network for image data
            self.net = net(_network_params)
            self.image_elements = torch.prod(torch.tensor(self.obs_shape[1:], dtype=torch.int)).item()
        else:
            # Concatenate the observation and explanation sizes for non-image data embedding
            input_size = input_shape[-1]
            # Embedding layer for continuous features of combined size
            self.input_embedding_layer = ContinuousFeatureJointLayer(d_model, input_size, explain_size)
            # Initialize the neural network for non-image data
            self.net = net(network_params)
        # Additional layers for non-linearity and final output transformation
        self.relu = nn.ReLU()
        self.final_layer = create_layer(d_model, output_size, act_fn=act_fn)

        # Apply initial weights
        self.apply(init_weights)
    
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
        if self.use_image:
            image_e = self._convert_explanation_to_image_shape(e)
            z = torch.cat([obs, image_e], dim = 1)
            y = self.net(z)
        else:
            z = self.input_embedding_layer(obs, e)
            y = self.net(z) if padding_mask is None else self.net(z, padding_mask)
            
        y = self.relu(y)
        y = self.final_layer(y)
        return y

    def _convert_explanation_to_image_shape(self, e):
        """ Convert the explanation vector to match the target image shape with the first dimension set to 1. """
        explain_shape = [1] + list(self.obs_shape[1:])  # Set first dim to 1, rest match target shape
        e1 = e.repeat(1, self.image_elements // self.explain_size)
        e2 = torch.zeros_like(e[:, :self.image_elements % self.explain_size])
        expanded_e = torch.cat([e1, e2], dim=-1)  # Repeat to match the volume of target shape
        expanded_e = expanded_e.view(-1, *explain_shape)  # Reshape explanation vector to the new explain_shape
        return expanded_e