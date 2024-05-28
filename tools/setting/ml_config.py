from tools.setting.ml_params import RESNET_COOPERATIVE_NETWORK, STYLEGAN_COOPERATIVE_NETWORK, GPT_COOPERATIVE_NETWORK, DEEPFM_COOPERATIVE_NETWORK, RESNET_BOTTLE_NECK_COOPERATIVE_NETWORK
from copy import deepcopy

def configure_model(model_name, params, obs_shape, condition_dim, z_dim):
    cooperative_network = None
    if model_name == 'stylegan':
        cooperative_network = STYLEGAN_COOPERATIVE_NETWORK
    elif model_name == 'resnet':
        cooperative_network = RESNET_COOPERATIVE_NETWORK
    elif model_name == 'resnet_bottle_neck':
        cooperative_network = RESNET_BOTTLE_NECK_COOPERATIVE_NETWORK
    elif model_name == 'gpt':
        cooperative_network = GPT_COOPERATIVE_NETWORK
    elif model_name == 'deepfm':
        cooperative_network = DEEPFM_COOPERATIVE_NETWORK
    else:
        raise ValueError(f"Model name '{model_name}' is not supported.")
        
    params.model_name = model_name
    params.obs_shape = obs_shape
    params.condition_dim = condition_dim    
    params.z_dim = z_dim    
    return cooperative_network, params

def configure_encoder_model(data_config, model_name, model_params):
    obs_shape = data_config.obs_shape
    if data_config.state_size is None:
        stoch_size, det_size = max(model_params.d_model//2, 1), max(model_params.d_model//2, 1)
        data_config.state_size = stoch_size + det_size
    else:
        stoch_size, det_size = data_config.state_size//2, data_config.state_size - data_config.state_size//2
    return configure_model(model_name, model_params, obs_shape, condition_dim=stoch_size, z_dim=det_size)
    
def configure_core_model(data_config, model_name, model_params):
    obs_shape = data_config.obs_shape if data_config.state_size is None else [data_config.state_size]
    if data_config.task_type == 'ordinal_regression':
        label_size = 1
    else:        
        label_size = data_config.label_size
    explain_size = max(model_params.d_model//2, 1) if data_config.explain_size is None else data_config.explain_size
    return configure_model(model_name, model_params, obs_shape, condition_dim=label_size, z_dim=explain_size)

def modify_network_params(network_params, attribute=None, value=None):
    """
    Modify a specific attribute value in the network parameters object.
    
    Args:
        network_params (object): An object containing network parameters.
        attribute (str): The attribute name to modify.
        value: The new value to assign to the specified attribute.

    Returns:
        object: A modified copy of network_params with the updated attribute value,
                or an unmodified copy if attribute or value is None.
    """
    # Create a deep copy of network parameters to ensure the original parameters remain unmodified
    copy_network_params = deepcopy(network_params)
    
    # Check if attribute or value is None and return the copy without modification
    if attribute is None or value is None:
        return copy_network_params
    
    if hasattr(copy_network_params, attribute):
        # Update the specified attribute with the new value
        setattr(copy_network_params, attribute, value)
    
    return copy_network_params