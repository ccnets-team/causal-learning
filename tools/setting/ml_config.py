import math
from tools.setting.ml_params import GPTModelParams, ImageModelParams

def configure_image_model(params, obs_shape):
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
    return params
