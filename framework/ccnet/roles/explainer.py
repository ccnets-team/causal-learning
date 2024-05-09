'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''

import torch.nn as nn
from nn.utils.init import init_weights, create_layer
from nn.utils.init import ContinuousFeatureJointLayer

class Explainer(nn.Module):
    """
    The Explainer module is designed to process inputs and make explanations based on the input data,
    whether it's tabular or image data. It adjusts its behavior based on the input type.

    Attributes:
        use_image (bool): Determines if the input data is image data based on the shape of the input.
        input_embedding_layer (nn.Module): Transforms continuous input features into an embedded space.
        relu (nn.Module): A ReLU activation layer to introduce non-linearity.
        final_layer (nn.Module): Final transformation layer to produce the desired output size.
        net (nn.Module): The main neural network module that processes the embedded or raw input.
    """

    def __init__(self, net, network_params, act_fn='none'):
        """
        Initializes the Explainer module with the specified network architecture and parameters.

        Parameters:
            net (callable): A callable that returns an nn.Module when passed network_params.
            network_params (object): Parameters specific to the neural network being used.
            input_shape (tuple): The shape of the input data.
            output_size (int): The size of the output tensor after final transformation.
            act_fn (str): The activation function name to use in the final layer (default 'none').
        """
        super(Explainer, self).__init__()
        
        input_shape, d_model, output_size = network_params.obs_shape, network_params.d_model, network_params.z_dim
        
        # Check if the input is image data based on the dimensionality
        self.use_image = len(input_shape) != 1
        
        # For non-image, tabular data:
        if not self.use_image:
            input_size = input_shape[-1]  # Size of the last dimension of the input
            # Embedding layer for continuous features
            self.input_embedding_layer = ContinuousFeatureJointLayer(d_model, input_size)

        # Initialize the main network module
        self.net = net(network_params)
                    
        # Activation layer
        self.relu = nn.ReLU(inplace=True)
        # Final layer to adjust to the required output size, with specified activation function
        self.final_layer = create_layer(d_model, output_size, act_fn=act_fn) 
        
        # Apply initial weights
        self.apply(init_weights)

    def forward(self, x, padding_mask=None):
        """
        Defines the forward pass of the Explainer module.

        Parameters:
            x (Tensor): The input data tensor.
            padding_mask (Tensor, optional): Optional padding mask to be used on input data.

        Returns:
            Tensor: The output tensor after processing through the network.
        """
        if self.use_image:
            e = self.net(x)
        else:
            x = self.input_embedding_layer(x)
            e = self.net(x) if padding_mask is None else self.net(x, padding_mask = padding_mask)

        e = self.relu(e)
        e = self.final_layer(e)
        return e
