from nn.gpt import GPT
from nn.custom_style_gan import Discriminator, ConditionalDiscriminator, ConditionalGenerator

class GPTParams:
    def __init__(self, num_layers=6, d_model=256, dropout=0.02):
        """
        Initialize parameters for a GPT model configuration.
        
        Args:
        - num_layers (int): Number of transformer layers in the GPT model.
        - d_model (int): The dimensionality of the model's embeddings and hidden layers.
        - dropout (float): Dropout rate to use between layers to prevent overfitting.
        """
        self.num_layers = num_layers
        self.d_model = d_model
        self.dropout = dropout

class EncodingParams:
    def __init__(self, num_layers=6, d_model=256, obs_shape=[], z_dim=None, condition_dim=None):
        """
        Parameters for encoding networks, possibly for use in models like VAEs or conditional GANs.
        
        Args:
        - num_layers (int): Number of layers in the encoding network.
        - d_model (int): Dimensionality of the encoding model's layers.
        - obs_shape (list): The shape of the input observations/data.
        - z_dim (int, optional): Dimensionality of the latent space.
        - condition_dim (int, optional): Dimensionality of any conditioning variables.
        """
        self.num_layers = num_layers
        self.d_model = d_model
        self.obs_shape = obs_shape
        self.z_dim = z_dim
        self.condition_dim = condition_dim

class ModelParameters:
    def __init__(self, num_layers=6, d_model=256, dropout=0.02, 
                 encoding_networks=[Discriminator, ConditionalDiscriminator, ConditionalGenerator]):
        """
        Comprehensive model parameters, combining core and encoding networks.
        
        Args:
        - num_layers (int): Number of layers for the core network (GPT).
        - d_model (int): Dimensionality of the core network's model.
        - dropout (float): Dropout rate for the core network.
        - encoding_networks (list): List of classes representing different encoding network types.
        """
        self.core_networks = [GPT, GPT, GPT]
        self.core_params = GPTParams(num_layers=num_layers, d_model=d_model, dropout=dropout)
        self.encoding_networks = encoding_networks
        self.encoding_params = EncodingParams(num_layers=num_layers, d_model= d_model)
    
class OptimizationParameters:
    def __init__(self, learning_rate=2e-4, decay_rate_100k=0.05, scheduler_type='exponential', clip_grad_range=None, max_grad_norm=1.0):
        """
        Parameters for optimization, including learning rate adjustments and gradient clipping.
        
        Args:
        - learning_rate (float): Initial learning rate for optimization.
        - decay_rate_100k (float): Rate at which the learning rate decays every 100,000 training steps.
        - scheduler_type (str): Type of scheduler for learning rate adjustment ('linear', 'exponential', or 'cyclic').
        - clip_grad_range (tuple, optional): Tuple specifying the minimum and maximum range for gradient clipping.
        - max_grad_norm (float): Maximum allowable L2 norm for gradients, to prevent gradient explosion.
        """
        self.learning_rate = learning_rate
        self.decay_rate_100k = decay_rate_100k
        self.scheduler_type = scheduler_type
        self.clip_grad_range = clip_grad_range
        self.max_grad_norm = max_grad_norm

class TrainingParameters:
    def __init__(self, num_epoch=100, max_iters=1e+6, batch_size=64, seq_len=1):
        """
        Initialize training parameters for machine learning models.
        
        Args:
        - num_epoch (int): Number of training epochs. One epoch represents a complete pass through the entire dataset.
        - total_iters (int): Total number of iterations or updates to the model during training.
        - batch_size (int): Number of samples to process in each batch during training.
        - seq_len (int): Length of the input sequences, applicable primarily in models dealing with sequence data.
        
        Note: Training will halt when either the total number of epochs ('num_epoch') or the total number of iterations
        ('total_iters') is reached, whichever comes first. This dual limit approach provides control over training duration and computational resources.
        """
        self.max_epoch = num_epoch
        self.max_iters = int(max_iters)  # Ensure total_iters is an integer to avoid type issues.
        self.batch_size = batch_size
        self.seq_len = seq_len

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