from nn.gpt import GPT
from nn.custom_style_gan import Discriminator, Generator
from nn.custom_resnet import ResNet_basic, transpose_ResNet_basic
from nn.custom_deepFM import ContinuousDeepFM
from nn.mlp import MLP
from nn.custom_super_net import SuperNet
from dataclasses import dataclass, field

GPT_COOPERATIVE_NETWORK = [GPT, GPT, GPT]
RESNET_COOPERATIVE_NETWORK = [ResNet_basic, ResNet_basic, transpose_ResNet_basic]
STYLEGAN_COOPERATIVE_NETWORK = [Discriminator, Discriminator, Generator]
DEEPFM_COOPERATIVE_NETWORK = [ContinuousDeepFM, ContinuousDeepFM, SuperNet]
MLP_COOPERATIVE_NETWORK = [MLP, MLP, MLP]

@dataclass
class ModelConfig:
    model_name: str
    
    num_layers: int = 6
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
class TrainingParameters:
    """
    Parameters defining the training configuration for machine learning models.
    
    Attributes:
        num_epoch (int): Number of training epochs. One epoch is a complete pass through the entire dataset.
        max_iters (int): Total number of iterations or updates to the model during training.
        batch_size (int): Number of samples to process in each batch during training.
        
    Note:
        Training will halt when either the total number of epochs ('num_epoch') or the total number of iterations
        ('max_iters') is reached, whichever comes first. This dual limit approach provides control over training duration and computational resources.
    """
    num_epoch: int = 100
    max_iters: int = 1_000_000
    batch_size: int = 64

@dataclass
class AlgorithmParameters:
    enable_diffusion : bool = False

@dataclass
class ModelParameters:
    """
    Comprehensive parameters defining core and encoding model configurations.
    
    Attributes:
        core_model (str): Identifier for the core model. A value of 'none' indicates no core model is used.
        encoder_model (str): Identifier for the encoder model. A value of 'none' indicates no encoder model is used.
        core_config (ModelConfig or None): Configuration object for the core model, if applicable.
        encoder_config (ModelConfig or None): Configuration object for the encoder model, if applicable.
    """
    core_model: str = 'gpt'
    encoder_model: str = 'resnet'
    core_config: ModelConfig = field(init=False)
    encoder_config: ModelConfig = field(init=False)

    def __post_init__(self):
        # Conditionally initialize the core_config
        if self.core_model.lower() != 'none':
            self.core_config = ModelConfig(model_name=self.core_model)
        else:
            self.core_config = None  # Properly handle 'none' to avoid creating a config

        # Conditionally initialize the encoder_config
        if self.encoder_model.lower() != 'none':
            self.encoder_config = ModelConfig(model_name=self.encoder_model)
        else:
            self.encoder_config = None  # Properly handle 'none' to avoid creating a config

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
    decay_rate_100k: float = 0.01
    scheduler_type: str = 'exponential'
    clip_grad_range: tuple = field(default=None)
    max_grad_norm: float = 1.0


class MLParameters:
    def __init__(self, 
                 training: TrainingParameters = None,
                 algorithm: AlgorithmParameters = None,
                 model: ModelParameters = None,
                 optimization: OptimizationParameters = None,
                 **kwargs):
        # Use kwargs to set up initial parameters, categorizing them for each parameter class
        training_kwargs = {k: v for k, v in kwargs.items() if k in TrainingParameters.__annotations__}
        algorithm_kwargs = {k: v for k, v in kwargs.items() if k in AlgorithmParameters.__annotations__}
        model_kwargs = {k: v for k, v in kwargs.items() if k in ModelParameters.__annotations__}
        optimization_kwargs = {k: v for k, v in kwargs.items() if k in OptimizationParameters.__annotations__}

        # Initialize ML parameters with filtered kwargs
        self.training = TrainingParameters(**training_kwargs) if training is None else training
        self.algorithm = AlgorithmParameters(**algorithm_kwargs) if algorithm is None else algorithm
        self.model = ModelParameters(**model_kwargs) if model is None else model
        self.optimization = OptimizationParameters(**optimization_kwargs) if optimization is None else optimization

    def __getattr__(self, name):
        # Check if the attribute is part of any of the parameter classes
        for param in [self.training, self.algorithm, self.model, self.optimization]:
            if hasattr(param, name):
                return getattr(param, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Set attribute if it's one of MLParameters' direct attributes
        if name in ["training", "algorithm", "model", "optimization"]:
            super().__setattr__(name, value)
        else:
            # Set attribute in one of the parameter classes
            for param in [self.training, self.algorithm, self.model, self.optimization]:
                if hasattr(param, name):
                    setattr(param, name, value)
                    return
            # If the attribute is not found in any of the parameter classes, set it as a new attribute of MLParameters
            super().__setattr__(name, value)

    def __iter__(self):
        yield from [self.training, self.algorithm, self.model, self.optimization]
