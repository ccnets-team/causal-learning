from nn.gpt import GPT
from nn.custom_style_gan import Discriminator, ConditionalDiscriminator, ConditionalGenerator

class GPTParameters:
    def __init__(self, num_layers=5, d_model=256, dropout=0.05):
        """
        Initialize a GPT network.	
        Args:
        - d_model (int): Dimension of the model.
        - num_layers (int): Number of layers in the network.
        - dropout (float): Dropout rate.
        """
        self.num_layers = num_layers
        self.d_model = d_model
        self.dropout = dropout

class EncodingParameters:
    def __init__(self, obs_shape = [], z_dim = None, condition_dim = None, num_layers=5, d_model=256):
        self.obs_shape = obs_shape
        self.z_dim = z_dim
        self.condition_dim = condition_dim
        self.num_layers = num_layers
        self.d_model = d_model
        
class TrainingParameters:
    def __init__(self, num_epoch=100, batch_size=32, seq_len=1):
        self.num_epoch = num_epoch  # Maximum number of steps for the exploration phase. This defines the period over which the exploration strategy is applied.
        self.batch_size = batch_size  # Number of samples processed before model update; larger batch size can lead to more stable but slower training.
        self.seq_len = seq_len
        self.total_iters = None
        # Note: Training begins only after the replay buffer is filled to its full capacity.

class ModelParameters:
    def __init__(self, num_layers=5, d_model=256, dropout=0.05, encoding_networks = [Discriminator, ConditionalDiscriminator, ConditionalGenerator]):
        self.core_networks = [GPT, GPT, GPT]
        self.core_params = GPTParameters(num_layers=num_layers, d_model=d_model, dropout=dropout)

        self.encoding_networks = encoding_networks
        self.encoding_params = EncodingParameters(num_layers=num_layers, d_model=d_model)
    
class OptimizationParameters:
    def __init__(self, learning_rate=2e-4, decay_rate_100k=0.01, scheduler_type='exponential', clip_grad_range=None, max_grad_norm=1.0): 
        self.learning_rate = learning_rate  # Learning rate for optimization algorithms, crucial for convergence.
        self.decay_rate_100k = decay_rate_100k  # Decay rate for the learning rate every 100k steps.
        self.scheduler_type = scheduler_type  # Type of learning rate scheduler: 'linear', 'exponential', or 'cyclic'.
        self.clip_grad_range = clip_grad_range  # Range for clipping gradients, preventing exploding gradients.
        self.max_grad_norm = max_grad_norm  # L2 norm threshold for gradient clipping to prevent exploding gradients.
        
class MLParameters:
    def __init__(self, 
                 training: TrainingParameters = None,
                 model: ModelParameters = None,
                 optimization: OptimizationParameters = None):
        # Initialize ML parameters
        self.training = TrainingParameters() if training is None else training
        self.model = ModelParameters() if model is None else model
        self.optimization = OptimizationParameters() if optimization is None else optimization
        self.selected_indices = []

    def __getattr__(self, name):
        # Check if the attribute is part of any of the parameter classes
        for param in [self.training, self.model, self.optimization]:
            if hasattr(param, name):
                return getattr(param, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Set attribute if it's one of MLParameters' direct attributes
        if name in ["training", "model", "optimization"]:
            super().__setattr__(name, value)
        else:
            # Set attribute in one of the parameter classes
            for param in [self.training, self.model, self.optimization]:
                if hasattr(param, name):
                    setattr(param, name, value)
                    return
            # If the attribute is not found in any of the parameter classes, set it as a new attribute of MLParameters
            super().__setattr__(name, value)

    def __iter__(self):
        yield from [self.training, self.model, self.optimization]