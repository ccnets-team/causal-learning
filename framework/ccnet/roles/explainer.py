'''
    COPYRIGHT (c) 2022. CCNets, Inc. All Rights reserved.
    Author:
        PARK, JunHo
'''

import torch.nn as nn
from nn.utils.init import init_weights
from nn.utils.joint_layer import JointLayer
from nn.utils.final_layer import FinalLayer
from tools.setting.ml_config import modify_network_params

class Explainer(nn.Module):
    """
    The Explainer module processes inputs and makes explanations based on the input data,
    whether it's tabular or image data. It adjusts its behavior based on the input type.

    Attributes:
        use_image (bool): Determines if the input data is image data based on the shape of the input.
        input_embedding_layer (nn.Module): Transforms continuous input features into an embedded space.
        final_layer (nn.Module): Final transformation layer to produce the desired output size.
        net (nn.Module): The main neural network module that processes the embedded or raw input.
    """

    def __init__(self, net, network_params, reset_pretrained, act_fn='none'):
        """
        Initializes the Explainer module with the specified network architecture and parameters.

        Parameters:
            net (callable): A callable that returns an nn.Module when passed network_params.
            network_params (object): Parameters specific to the neural network being used.
            act_fn (str): The activation function name to use in the final layer (default 'none').
        """
        super(Explainer, self).__init__()
        
        explainer_params = modify_network_params(network_params, None)
        input_shape, d_model, output_size = (explainer_params.obs_shape, 
                                             explainer_params.d_model, 
                                             explainer_params.z_dim)

        self.use_image = len(input_shape) != 1

        if not self.use_image:
            self.joint_layer = JointLayer(d_model, input_shape)

        self.net = net(explainer_params)
        self.final_layer = FinalLayer(d_model, output_size, first_act_fn='relu', last_act_fn=act_fn)
                
        self.apply(lambda module: init_weights(module, reset_pretrained))

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
            x = self.joint_layer(x) # Joint Layer for single input
            e = self.net(x) if padding_mask is None else self.net(x, padding_mask=padding_mask)

        e = self.final_layer(e)
        return e
