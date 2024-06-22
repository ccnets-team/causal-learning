from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """
    Configuration for defining model settings in CCNet.

    Attributes:
        model_name (str): Identifier for the core model. Default is 'gpt'. Determines internal configurations.
        num_layers (int): Number of layers in the model.
        d_model (int): Dimensionality of the model's hidden layers.
        dropout (float): Dropout rate for regularization.
    """
    model_name: str = 'gpt'
    num_layers: int = 5
    d_model: int = 256
    dropout: float = 0.05
    use_seq_input: bool = False
    
    def __post_init__(self):
        if self.model_name == 'gpt':
            self.use_seq_input = True

    def __repr__(self):
        return (f"ModelConfig(model_name='{self.model_name}', num_layers={self.num_layers}, "
                f"d_model={self.d_model}, dropout={self.dropout})\n")

@dataclass
class TrainConfig:
    """
    Configuration for defining training settings.

    Attributes:
        num_epoch (int): Number of training epochs.
        batch_size (int): Number of samples per training batch.
        max_seq_len (int, optional): Maximum sequence length for training. Default is None.
        min_seq_len (int, optional): Minimum sequence length for training. Default is None.
        error_function (str): Error function used for training ('mae' or 'mse').
    """
    num_epoch: int = 100
    batch_size: int = 64
    max_seq_len: int = None
    min_seq_len: int = None
    error_function: str = 'mse'

    def __repr__(self):
        max_seq_repr = f", max_seq_len={self.max_seq_len}" if self.max_seq_len is not None else ""
        min_seq_repr = f", min_seq_len={self.min_seq_len}" if self.min_seq_len is not None else ""
        return (f"TrainConfig(num_epoch={self.num_epoch}, "
                f"batch_size={self.batch_size}{max_seq_repr}{min_seq_repr}, error_function='{self.error_function}')\n")

@dataclass
class OptimConfig:
    """
    Configuration for optimizing the training process.

    Attributes:
        learning_rate (float): Initial learning rate for optimization.
        decay_rate_100k (float): Rate at which the learning rate decays every 100,000 steps.
        scheduler_type (str): Type of learning rate scheduler ('linear', 'exponential', 'cyclic').
        clip_grad_range (tuple, optional): Range for gradient clipping (min, max). Default is None.
        max_grad_norm (float): Maximum allowable gradient norm to prevent gradient explosion.
    """
    learning_rate: float = 1e-3
    decay_rate_100k: float = 0.05
    scheduler_type: str = 'exponential'
    clip_grad_range: tuple = field(default=None)
    max_grad_norm: float = 1.0

    def __repr__(self):
        clip_grad_repr = f", clip_grad_range={self.clip_grad_range}" if self.clip_grad_range is not None else ""
        max_grad_norm_repr = f", max_grad_norm={self.max_grad_norm}" if self.max_grad_norm is not None else ""
        return (f"OptimConfig(learning_rate={self.learning_rate}, "
                f"decay_rate_100k={self.decay_rate_100k}, scheduler_type={self.scheduler_type}"
                f"{clip_grad_repr}{max_grad_norm_repr})\n")

class MLConfig:
    def __init__(self, 
                 model: ModelConfig = None,
                 training: TrainConfig = None,
                 optimization: OptimConfig = None,
                 **kwargs):
        def filter_kwargs(cls):
            return {k: v for k, v in kwargs.items() if k in cls.__annotations__}

        self.model = model or ModelConfig(**filter_kwargs(ModelConfig))
        self.training = training or TrainConfig(**filter_kwargs(TrainConfig))
        self.optimization = optimization or OptimConfig(**filter_kwargs(OptimConfig))
        
    def __getattr__(self, name):
        # Check if the attribute is part of any of the parameter classes
        for param in [self.model, self.training, self.optimization]:
            if hasattr(param, name):
                return getattr(param, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Set attribute if it's one of MLConfig' direct attributes
        if name in ["model", "training", "optimization"]:
            super().__setattr__(name, value)
        else:
            # Set attribute in one of the parameter classes
            for param in [self.model, self.training, self.optimization]:
                if hasattr(param, name):
                    setattr(param, name, value)
                    return
            # If the attribute is not found in any of the parameter classes, set it as a new attribute of MLConfig
            super().__setattr__(name, value)

    def __iter__(self):
        yield from [self.model, self.training, self.optimization]