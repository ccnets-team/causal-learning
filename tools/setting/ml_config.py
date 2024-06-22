from tools.setting.ml_params import GPT_COOPERATIVE_NETWORK
from tools.setting.ml_params import MLP_COOPERATIVE_NETWORK, TABNET_COOPERATIVE_NETWORK
from tools.setting.ml_params import RESNET18_COOPERATIVE_NETWORK, RESNET34_COOPERATIVE_NETWORK, RESNET50_COOPERATIVE_NETWORK
from copy import deepcopy

def configure_networks(model_name):
    networks = None
    if  model_name == 'gpt':
        networks = GPT_COOPERATIVE_NETWORK
    elif model_name == 'mlp':
        networks = MLP_COOPERATIVE_NETWORK
    elif model_name == 'tabnet':
        networks = TABNET_COOPERATIVE_NETWORK
    elif model_name == 'resnet':
        networks = RESNET18_COOPERATIVE_NETWORK
    elif model_name == 'resnet18':
        networks = RESNET18_COOPERATIVE_NETWORK
    elif model_name == 'resnet34':
        networks = RESNET34_COOPERATIVE_NETWORK
    elif model_name == 'resnet50':
        networks = RESNET50_COOPERATIVE_NETWORK
    else:
        raise ValueError(f"Model name '{model_name}' is not supported.")

    return networks

def configure_ccnet(model_params, data_config):
    """
    Configures the Causal Cooperative Network and its parameters.

    Args:
        model_params (ModelParameters): Model parameters including model name and configuration details.
        data_config (DataConfig): Data configuration including observational shape, task type, and label sizes.

    Returns:
        tuple: A tuple containing the configured cooperative network and updated model parameters.
    """
    obs_shape = data_config.obs_shape
    label_size = data_config.label_size
    task_type = data_config.task_type
    explain_size = data_config.explain_size

    if task_type in ['ordinal_regression', 'binary_classification']:
        label_size = 1
    model_params.y_dim = label_size    

    if explain_size is None:
        if len(obs_shape) != 1:
            explain_size = max(model_params.d_model // 2, 1)
        else:
            explain_size = int(max(round((obs_shape[-1] - label_size) / 2), 1))
    data_config.explain_size = explain_size
    model_params.e_dim = explain_size    
    model_params.obs_shape = obs_shape    

    networks = configure_networks(model_params.model_name)
    
    return networks, model_params

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

def determine_max_iters_and_epoch(ml_params):
    """
    The training duration follows based on the shortest ends either 'max_iters' or 'num_epoch', unless one is missing.
    
    Parameters:
    ml_params (object): Contains 'num_epoch', 'max_iters', and 'batch_size' under 'training'.

    Raises:
    ValueError: If both 'num_epoch' and 'max_iters' are None.
    """    
    if ml_params.num_epoch is None:
        ml_params.num_epoch = ml_params.training.max_iters // ml_params.training.batch_size
    elif ml_params.training.max_iters is None:
        ml_params.training.max_iters = ml_params.num_epoch * ml_params.training.batch_size
    elif ml_params.num_epoch is not None and ml_params.training.max_iters is not None:
        epoch_iters = ml_params.num_epoch * ml_params.training.batch_size
        if epoch_iters > ml_params.training.max_iters:
            ml_params.num_epoch = ml_params.training.max_iters // ml_params.training.batch_size
        else:
            ml_params.training.max_iters = epoch_iters
    else:
        raise ValueError("Both 'num_epoch' and 'max_iters' cannot be set. Please set only one of them.")