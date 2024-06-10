'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho@ccnets.org
'''

import torch
import torch.nn as nn
from nn.utils.init import init_weights
from nn.utils.joint_layer import JointLayer
from nn.utils.final_layer import FinalLayer
from framework.utils.ccnet_utils import convert_explanation_to_image_shape, extend_obs_shape_channel
from tools.setting.ml_config import modify_network_params

class Reasoner(nn.Module):
    """
    The Reasoner module processes and reasons about data, integrating explanations into its predictions,
    for both image and non-image data.

    Attributes:
        use_image (bool): Flag to determine if the input data is image-based.
        obs_shape (tuple): The shape of the input data.
        explain_size (int): The size of the explanation data to be integrated.
        input_embedding_layer (nn.Module): Embedding layer for non-image data.
        image_elements (int): Total elements in an image if using image data.
        net (nn.Module): The main neural network, adaptable for both image and non-image inputs.
        final_layer (nn.Module): The final transformation layer to produce the desired output.
    """

    def __init__(self, net, network_params, reset_pretrained, act_fn='none'):
        """
        Initializes the Reasoner module with network architecture and parameters.

        Parameters:
            net (callable): A callable that returns an nn.Module when passed network_params.
            network_params (object): Parameters specific to the neural network being used.
            act_fn (str): The activation function name to use in the final layer (default 'none').
        """
        super(Reasoner, self).__init__()
        
        input_shape, d_model, explain_size, output_size = (network_params.obs_shape, 
                                                           network_params.d_model, 
                                                           network_params.z_dim, 
                                                           network_params.condition_dim)
        
        self.use_image = len(input_shape) != 1
        self.obs_shape = input_shape
        self.explain_size = explain_size

        if self.use_image:
            extended_obs_shape = extend_obs_shape_channel(input_shape)
            reasoner_params = modify_network_params(network_params, 'obs_shape', extended_obs_shape)
            self.image_elements = torch.prod(torch.tensor(extended_obs_shape[1:], dtype=torch.int)).item()
        else:
            reasoner_params = modify_network_params(network_params)
            self.joint_layer = JointLayer(d_model, input_shape, explain_size)

        self.net = net(reasoner_params)
        self.final_layer = FinalLayer(d_model, output_size, first_act_fn='relu', last_act_fn=act_fn)
        self.apply(lambda module: init_weights(module, reset_pretrained))
    
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
            z = torch.cat([obs, image_e], dim=1)
        else:
            z = self.joint_layer(obs, e)
        y = self.net(z) if padding_mask is None else self.net(z, padding_mask=padding_mask)
        y = self.final_layer(y)
        return y

    def _convert_explanation_to_image_shape(self, explanation):
        return convert_explanation_to_image_shape(explanation=explanation, 
                                                  image_shape=self.obs_shape, 
                                                  explain_size=self.explain_size, 
                                                  image_elements=self.image_elements)