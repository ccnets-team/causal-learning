from tools.config.ccnet_config import GPT_COOPERATIVE_NETWORK
from tools.config.ccnet_config import MLP_COOPERATIVE_NETWORK, TABNET_COOPERATIVE_NETWORK
from tools.config.ccnet_config import RESNET18_COOPERATIVE_NETWORK, RESNET34_COOPERATIVE_NETWORK, RESNET50_COOPERATIVE_NETWORK
from tools.config.ccnet_config import CCNetConfig

def configure_networks(model_config):
    model_name = model_config.model_name
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

def configure_ccnet_config(data_config, model_config):
    """
    Configures CCNet settings based on the provided data and model configurations.

    Args:
        data_config (DataConfig): Configuration including observational shape, task type, and label sizes.
        model_config (ModelConfig): Configuration including model name and details.

    Returns:
        CCNetConfig: Configured settings for CCNet based on data and model configurations.
    """
    # Extract necessary attributes from data_config
    obs_shape = data_config.obs_shape
    label_size = data_config.label_size
    task_type = data_config.task_type
    explain_size = data_config.explain_size
    label_scale = data_config.label_scale

    # Adjust label_size for specific task types
    if task_type in ['ordinal_regression', 'binary_classification']:
        label_size = 1

    # Calculate explain_size if not provided
    if explain_size is None:
        if len(obs_shape) != 1:
            explain_size = max(model_config.d_model // 2, 1)
        else:
            explain_size = int(max(round((obs_shape[-1] - label_size) / 2), 1))

    # Create and configure CCNetConfig
    ccnet_config = CCNetConfig(
        model_name=model_config.model_name,
        num_layers=model_config.num_layers,
        d_model=model_config.d_model,
        dropout=model_config.dropout,
        use_seq_input=model_config.use_seq_input,
        obs_shape=obs_shape,
        y_dim=label_size,
        e_dim=explain_size,
        task_type=task_type,
        y_scale=label_scale
    )

    return ccnet_config
