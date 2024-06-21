from nn.gpt import GPT
from nn.resnet import ResNet18, ResNet34, ResNet50
from nn.transpose_resnet import TransposeResnet
from nn.mlp import MLP
from nn.tabnet import EncoderTabNet, DecoderTabNet
from dataclasses import dataclass, field
from typing import Generator, Any
import torch

GPT_COOPERATIVE_NETWORK = [GPT, GPT, GPT]

RESNET18_COOPERATIVE_NETWORK = [ResNet18, ResNet18, TransposeResnet]
RESNET34_COOPERATIVE_NETWORK = [ResNet34, ResNet34, TransposeResnet]
RESNET50_COOPERATIVE_NETWORK = [ResNet50, ResNet50, TransposeResnet]

MLP_COOPERATIVE_NETWORK = [MLP, MLP, MLP]
TABNET_COOPERATIVE_NETWORK = [EncoderTabNet, EncoderTabNet, DecoderTabNet]

@dataclass
class BaseNetworkConfig:
    num_layers: int = 5
    d_model: int = 256
    dropout: float = 0.05
    obs_shape: list = field(default_factory=list)
    reset_pretrained: bool = False

    def reset(self):
        """Reset all attributes to their default values."""
        self.num_layers = 0
        self.d_model = 0
        self.dropout = 0
        self.obs_shape = []
        self.reset_pretrained = False
        self.device = None

    def apply_config(self, config_gen: Generator):
        """
        Apply settings from a generator.
        
        Args:
            config_gen (generator): Generator yielding key-value pairs.
        """
        for key, value in config_gen:
            setattr(self, key, value)

    def config_generator(self):
        """
        Generator to yield key-value pairs from a BaseNetworkConfig instance.
        """
        for key, value in self.__dict__.items():
            yield key, value
    
class NetworkConfig(BaseNetworkConfig):
    def __init__(self, base_network_config: BaseNetworkConfig, network_role_name: str, input_shape: Any, output_shape: Any, act_fn: str):
        super().__init__()
        self.apply_config(base_network_config.config_generator())  # Call the generator

        self.network_role_name = network_role_name
        self.input_shape = input_shape if isinstance(input_shape, (list, torch.Size, tuple)) else [input_shape]
        self.output_shape = output_shape if isinstance(output_shape, (list, torch.Size, tuple)) else [output_shape]
        self.act_fn = act_fn
                    
@dataclass
class CCNetConfig(BaseNetworkConfig):
    network_name: str = ''
    y_dim: int = None
    e_dim: int = None
    device: torch.device = None

    def __post_init__(self):
        if self.network_name.lower() == 'none':
            self.reset()
        
@dataclass
class ModelParameters:
    """
    Comprehensive parameters defining core and encoding model configurations.
    
    Attributes:
        ccnet_network (str): Identifier for the core model. A value of 'none' indicates no core model is used.
        ccnet_config (CCNetConfig or None): Configuration object for the core model, if applicable.
    """
    ccnet_network: str = 'gpt'
    ccnet_config: CCNetConfig = field(init=False)

    def __post_init__(self):
        # Conditionally initialize the ccnet_config
        if self.ccnet_network.lower() != 'none':
            self.ccnet_config = CCNetConfig(network_name=self.ccnet_network)
        else:
            self.ccnet_config = None  # Properly handle 'none' to avoid creating a config

    def __repr__(self):
        return (f"ModelParameters(ccnet_network={self.ccnet_network}\n")
        
@dataclass
class TrainingParameters:
    """
    Parameters defining the training configuration for machine learning models.
    
    Attributes:
        num_epoch (int): Number of training epochs. One epoch is a complete pass through the entire dataset.
        max_iters (int): Total number of iterations or updates to the model during training.
        batch_size (int): Number of samples to process in each batch during training.
        max_seq_len (int): Maximum sequence length for training.
        min_seq_len (int): Minimum sequence length for training.
        
    Note:
        Training will halt when either the total number of epochs ('num_epoch') or the total number of iterations
        ('max_iters') is reached, whichever comes first. This dual limit approach provides control over training duration and computational resources.
    """
    num_epoch: int = 100
    max_iters: int = 100_000
    batch_size: int = 64
    max_seq_len: int = None
    min_seq_len: int = None

    def __repr__(self):
        max_seq_repr = f", max_seq_len={self.max_seq_len}" if self.max_seq_len is not None else ""
        min_seq_repr = f", min_seq_len={self.min_seq_len}" if self.min_seq_len is not None else ""
        return (f"TrainingParameters(num_epoch={self.num_epoch}, max_iters={self.max_iters}, "
                f"batch_size={self.batch_size}{max_seq_repr}{min_seq_repr}\n")

@dataclass
class OptimizationParameters:
    """
    Parameters for optimizing the machine learning model training process.
    
    Attributes:
        learning_rate (float): Initial learning rate for optimization.
        decay_rate_100k (float): Rate at which the learning rate decays every 100,000 training steps.
        scheduler_type (str): Type of scheduler for learning rate adjustment ('linear', 'exponential', or 'cyclic').
        clip_grad_range (tuple, optional): Tuple specifying the minimum and maximum range for gradient clipping.
        max_grad_norm (float): Maximum allowable L2 norm for gradients to prevent gradient explosion.
    """
    learning_rate: float = 1e-3
    decay_rate_100k: float = 0.05
    scheduler_type: str = 'exponential'
    clip_grad_range: tuple = field(default=None)
    max_grad_norm: float = 1.0

    def __repr__(self):
        clip_grad_repr = f", clip_grad_range={self.clip_grad_range}" if self.clip_grad_range is not None else ""
        max_grad_norm = f", max_grad_norm={self.max_grad_norm}" if self.max_grad_norm is not None else ""
        return (f"OptimizationParameters(learning_rate={self.learning_rate}, "
                f"decay_rate_100k={self.decay_rate_100k}, scheduler_type={self.scheduler_type}"
                f"{clip_grad_repr}{max_grad_norm}\n")
        
@dataclass
class AlgorithmParameters:
    """
    Parameters defining the algorithm configuration for machine learning models.
    
    Attributes:
        reset_pretrained (bool): Determines if pretrained models are used for the ccnet network. At least one network in the ccnet uses a pretrained model.
        error_function (str): Error function used for the cooperative network (ccnet) which includes explainer, reasoner, and producer networks.
                              Two options are available: 'mae' (Mean Absolute Error) or 'mse' (Mean Squared Error).
                              'mae' is good for general cases or outliers, while 'mse' is suitable for generation tasks.
    """
    reset_pretrained: bool = False
    error_function: str = 'mse'

    def __repr__(self):
        return (f"AlgorithmParameters(reset_pretrained={self.reset_pretrained}, error_function={self.error_function})\n")

class MLParameters:
    def __init__(self, 
                 model: ModelParameters = None,
                 training: TrainingParameters = None,
                 optimization: OptimizationParameters = None,
                 algorithm: AlgorithmParameters = None,
                 **kwargs):
        def filter_kwargs(cls):
            return {k: v for k, v in kwargs.items() if k in cls.__annotations__}

        self.model = model or ModelParameters(**filter_kwargs(ModelParameters))
        self.training = training or TrainingParameters(**filter_kwargs(TrainingParameters))
        self.optimization = optimization or OptimizationParameters(**filter_kwargs(OptimizationParameters))
        self.algorithm = algorithm or AlgorithmParameters(**filter_kwargs(AlgorithmParameters))
        self.ml_param_list = [self.model, self.training, self.optimization, self.algorithm]
        
    def __getattr__(self, name):
        # Check if the attribute is part of any of the parameter classes
        for param in self.ml_param_list:
            if hasattr(param, name):
                return getattr(param, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Set attribute if it's one of MLParameters' direct attributes
        if name in ["model", "training", "optimization", "algorithm"]:
            super().__setattr__(name, value)
        else:
            # Set attribute in one of the parameter classes
            for param in [self.model, self.training, self.optimization, self.algorithm]:
                if hasattr(param, name):
                    setattr(param, name, value)
                    return
            # If the attribute is not found in any of the parameter classes, set it as a new attribute of MLParameters
            super().__setattr__(name, value)

    def __iter__(self):
        yield from self.ml_param_list