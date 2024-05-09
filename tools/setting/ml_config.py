from tools.setting.ml_params import RESNET_COOPERATIVE_NETWORKS, STYLEGAN_COOPERATIVE_NETWORKS, GPT_COOPERATIVE_NETWORKS
from copy import deepcopy

def configure_model(model_name, params, obs_shape, condition_dim, z_dim):
    networks = None
    if model_name == 'stylegan':
        networks = STYLEGAN_COOPERATIVE_NETWORKS
    elif model_name == 'resnet':
        networks = RESNET_COOPERATIVE_NETWORKS
    elif model_name == 'gpt':
        networks = GPT_COOPERATIVE_NETWORKS
    params.model_name = model_name
    params.obs_shape = obs_shape
    params.condition_dim = condition_dim    
    params.z_dim = z_dim    
    return networks, params

def configure_encoder_model(data_config, model_name, model_params):
    obs_shape = data_config.obs_shape
    stoch_size, det_size = max(model_params.d_model//2, 1), max(model_params.d_model//2, 1)
    if data_config.state_size is None:
        data_config.state_size = stoch_size + det_size
    return configure_model(model_name, model_params, obs_shape, condition_dim=stoch_size, z_dim=det_size)
    
def configure_core_model(data_config, model_name, model_params):
    obs_shape = data_config.obs_shape if data_config.state_size is None else [data_config.state_size]
    label_size = data_config.label_size
    explain_size = max(model_params.d_model//2, 1) if data_config.explain_size is None else data_config.explain_size
    return configure_model(model_name, model_params, obs_shape, condition_dim=label_size, z_dim=explain_size)

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

def modify_attribute_value(network_params, attribute, value):
    """
    Modify a specific attribute value in the network parameters object.
    
    Args:
        network_params (object): An object containing network parameters.
        attribute (str): The attribute name to modify.
        value: The new value to assign to the specified attribute.

    Returns:
        object: A modified copy of network_params with the updated attribute value.
    """
    # Create a deep copy of network parameters to ensure the original parameters remain unmodified
    copy_network_params = deepcopy(network_params)
    
    if hasattr(copy_network_params, attribute):
        # Update the specified attribute with the new value
        setattr(copy_network_params, attribute, value)
    
    return copy_network_params