import math
from tools.setting.ml_params import GPTModelParams, ImageModelParams, RESNET_COOPERATIVE_NETWORKS, STYLEGAN_COOPERATIVE_NETWORKS, GPT_COOPERATIVE_NETWORKS
from copy import deepcopy

def configure_image_model_params(params, obs_shape, condition_dim, z_dim):
    if not isinstance(params, ImageModelParams):
        return params
    
    """ Configure parameters for an image model based on image dimensions. """
    num_channels, h, w = obs_shape  # Assume `obs_shape` is available directly
    min_num_layers = int(math.log2(min(h, w)))
    num_layers = min(params.num_layers, min_num_layers)

    # Adjust `d_model` based on the effective number of layers
    adjusted_d_model = params.d_model // (2 ** (params.num_layers - num_layers))
    
    params.d_model = adjusted_d_model
    params.num_layers = num_layers
    params.obs_shape = obs_shape
    params.condition_dim = condition_dim    
    params.z_dim = z_dim
    return params

def configure_gpt_model_params(params, obs_shape, condition_dim, z_dim):
    if not isinstance(params, GPTModelParams):
        return params
    
    params.obs_shape = obs_shape
    params.condition_dim = condition_dim    
    params.z_dim = z_dim
    return params

def configure_model(model_name, params, obs_shape, condition_dim, z_dim):
    networks = None
    if model_name == 'stylegan':
        networks = STYLEGAN_COOPERATIVE_NETWORKS
    elif model_name == 'resnet':
        networks = RESNET_COOPERATIVE_NETWORKS
    elif model_name == 'gpt':
        networks = GPT_COOPERATIVE_NETWORKS
    if isinstance(params, GPTModelParams):
        return networks, configure_gpt_model_params(params, obs_shape, condition_dim, z_dim)
    elif isinstance(params, ImageModelParams):
        return networks, configure_image_model_params(params, obs_shape, condition_dim, z_dim)

def extend_obs_shape_channel(network_params):
    """
    Prepare and return modified network parameters for image data processing by adding an extra channel.
    
    Args:
        network_params (object): An object containing network parameters,
                                 specifically including the 'obs_shape' attribute which is expected 
                                 to be a tuple or list representing the dimensions of the input data.

    Returns:
        object: A modified copy of network_params with the incremented channel dimension in 'obs_shape'.
    """
    # Create a deep copy of network parameters to ensure the original parameters remain unmodified
    copy_network_params = deepcopy(network_params)
    
    # Ensure 'obs_shape' is mutable and increment the first dimension (channel count)
    copy_network_params.obs_shape = list(copy_network_params.obs_shape)
    copy_network_params.obs_shape[0] += 1
    
    return copy_network_params