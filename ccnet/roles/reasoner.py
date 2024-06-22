'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho@ccnets.org
'''

import torch
import torch.nn as nn
from nn.utils.init_layer import init_weights
from nn.utils.joint_layer import JointLayer
from ccnet.utils import extend_obs_shape_channel, convert_explanation_to_image_shape
from tools.setting.ccnet_config import CooperativeNetworkConfig

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

    def __init__(self, net, network_params, act_fn='none'):
        """
        Initializes the Reasoner module with network architecture and parameters.

        Parameters:
            net (callable): A callable that returns an nn.Module when passed network_params.
            network_params (object): Parameters specific to the neural network being used.
            act_fn (str): The activation function name to use in the final layer (default 'none').
        """
        super(Reasoner, self).__init__()

        self.__model_name = self._get_name()

        self.embedding_layer, reasoner_config = self._create_embedding_layer(network_params, act_fn)

        self.net = net(reasoner_config)
        
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
        z = self.embedding_layer(obs, e)
        y = self.net(z, padding_mask=padding_mask)
        return y

    def _create_embedding_layer(self, network_params, act_fn):
        """
        Creates the embedding layer based on the input data type and modifies network parameters accordingly.

        This function determines whether the input data is image-based or non-image-based, then sets up the
        appropriate embedding layer. For image data, it merges observations and reshaped explanation data.
        For non-image data, it uses a `JointLayer` to integrate the observation and explanation data directly.

        Parameters:
            network_params (object): Contains the parameters specific to the neural network, including
                                    input shape, model dimensions, and explanation size.

        Returns:
            function or nn.Module: The embedding layer function for image data, or an `nn.Module` for non-image data.
            object: Modified network parameters reflecting changes based on the input data type.
        """
        input_shape = network_params.obs_shape
        d_model = network_params.d_model
        explain_size = network_params.e_dim
        output_size = network_params.y_dim

        if len(input_shape) != 1:  # Handle image data
            extended_obs_shape = extend_obs_shape_channel(input_shape)
            image_elements = torch.prod(torch.tensor(extended_obs_shape[1:], dtype=torch.int)).item()
            reasoner_config = CooperativeNetworkConfig(network_params, self.__model_name, extended_obs_shape, output_size, act_fn)

            def embedding_layer_with_image(obs, e):
                image_e = convert_explanation_to_image_shape(
                    explanation=e,
                    image_shape=input_shape,
                    explain_size=explain_size,
                    image_elements=image_elements
                )
                return torch.cat([obs, image_e], dim=1)

            return embedding_layer_with_image, reasoner_config

        else:  # Handle non-image data
            embedding_layer = JointLayer(self.__model_name, d_model, input_shape, explain_size)
            reasoner_config = CooperativeNetworkConfig(network_params, self.__model_name, d_model, output_size, act_fn)
            return embedding_layer, reasoner_config