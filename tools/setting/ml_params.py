from nn.gpt import GPT
from nn.custom_style_gan import Discriminator, Generator
from nn.resnet import ResNet18, ResNet34, ResNet50
from nn.transpose_resnet import TransposeResnet
from nn.mlp import MLP
from nn.tabnet import TabNet
from dataclasses import dataclass, field

GPT_COOPERATIVE_NETWORK = [GPT, GPT, GPT]

RESNET18_COOPERATIVE_NETWORK = [ResNet18, ResNet18, TransposeResnet]
RESNET34_COOPERATIVE_NETWORK = [ResNet34, ResNet34, TransposeResnet]
RESNET50_COOPERATIVE_NETWORK = [ResNet50, ResNet50, TransposeResnet]

STYLEGAN_COOPERATIVE_NETWORK = [Discriminator, Discriminator, Generator]

MLP_COOPERATIVE_NETWORK = [MLP, MLP, MLP]
TABNET_COOPERATIVE_NETWORK = [TabNet, TabNet, TabNet]

@dataclass
class ModelConfig:
    model_name: str
    
    num_layers: int = 5
    d_model: int = 256
    dropout: float = 0.05
    
    obs_shape: list = field(default_factory=list)
    condition_dim: int = None
    z_dim: int = None

    def __post_init__(self):
        if self.model_name.lower() == 'none':
            # Handle the case where no model is required
            self.num_layers = 0
            self.d_model = 0
            self.dropout = 0
            self.obs_shape = []
            self.condition_dim = None
            self.z_dim = None
    
@dataclass
class ModelParameters:
    """
    Comprehensive parameters defining core and encoding model configurations.
    
    Attributes:
        ccnet_network (str): Identifier for the core model. A value of 'none' indicates no core model is used.
        encoder_network (str): Identifier for the encoder model. A value of 'none' indicates no encoder model is used.
        ccnet_config (ModelConfig or None): Configuration object for the core model, if applicable.
        encoder_config (ModelConfig or None): Configuration object for the encoder model, if applicable.
    """
    ccnet_network: str = 'gpt'
    encoder_network: str = 'none'
    ccnet_config: ModelConfig = field(init=False)
    encoder_config: ModelConfig = field(init=False)

    def __post_init__(self):
        # Conditionally initialize the ccnet_config
        if self.ccnet_network.lower() != 'none':
            self.ccnet_config = ModelConfig(model_name=self.ccnet_network)
        else:
            self.ccnet_config = None  # Properly handle 'none' to avoid creating a config

        # Conditionally initialize the encoder_config
        if self.encoder_network.lower() != 'none':
            self.encoder_config = ModelConfig(model_name=self.encoder_network)
        else:
            self.encoder_config = None  # Properly handle 'none' to avoid creating a config

    def __repr__(self):
        return (f"ModelParameters(ccnet_network={self.ccnet_network}, encoder_network={self.encoder_network}\n")
        
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
    learning_rate: float = 2e-4
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
        enable_diffusion (bool): Flag to enable the diffusion model, which is combined with the NoiseDiffuser class.
        reset_pretrained (bool): Determines if pretrained models are used for the ccnet network. At least one network in the ccnet uses a pretrained model.
        error_function (str): Error function used for the cooperative network (ccnet) which includes explainer, reasoner, and producer networks.
                              Two options are available: 'mae' (Mean Absolute Error) or 'mse' (Mean Squared Error).
                              'mae' is good for general cases or outliers, while 'mse' is suitable for generation tasks.
    """
    enable_diffusion: bool = False
    reset_pretrained: bool = False
    error_function: str = 'mse'

    def __repr__(self):
        return (f"AlgorithmParameters(enable_diffusion={self.enable_diffusion}, "
                f"reset_pretrained={self.reset_pretrained}, error_function={self.error_function})\n")

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

    def __repr__(self):
        components = []
        for param in self.ml_param_list:
            if param is not None:
                components.append(repr(param))
        return "\n".join(components)   
        
    def __iter__(self):
        yield from self.ml_param_list

