'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho@ccnets.org
'''

import torch
import torch.nn as nn
from nn.utils.init import init_weights, create_layer
from nn.utils.init import ContinuousFeatureJointLayer

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
        self.use_image = len(input_shape) != 1

        if self.use_image:
            # Initialize the neural network for image data
            self.net = net(network_params)
        else:
            # Concatenate the observation and explanation sizes for non-image data embedding
            input_size = input_shape[-1]
            # Embedding layer for continuous features of combined size
            self.input_embedding_layer = ContinuousFeatureJointLayer(input_size, explain_size, d_model)
            # Initialize the neural network for non-image data
            self.net = net(network_params)
        # Additional layers for non-linearity and final output transformation
        self.relu = nn.ReLU(inplace=True)
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
            y = self.net(obs, e)
        else:
            x = self.input_embedding_layer(obs, e)
            y = self.net(x) if padding_mask is None else self.net(x, padding_mask)
            
        y = self.relu(y)
        y = self.final_layer(y)
        return y
