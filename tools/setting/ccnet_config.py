from nn.gpt import GPT
from nn.resnet import ResNet18, ResNet34, ResNet50
from nn.resnet import TransposeResnet
from nn.mlp import MLP
from nn.tabnet import EncoderTabNet, DecoderTabNet
from dataclasses import dataclass, field
from typing import Generator, Any, List
import torch

RESNET18_COOPERATIVE_NETWORK = [ResNet18, ResNet18, TransposeResnet]
RESNET34_COOPERATIVE_NETWORK = [ResNet34, ResNet34, TransposeResnet]
RESNET50_COOPERATIVE_NETWORK = [ResNet50, ResNet50, TransposeResnet]

GPT_COOPERATIVE_NETWORK = [GPT, GPT, GPT]
MLP_COOPERATIVE_NETWORK = [MLP, MLP, MLP]
TABNET_COOPERATIVE_NETWORK = [EncoderTabNet, EncoderTabNet, DecoderTabNet]

@dataclass
class NetworkConfig:
    num_layers: int = 5
    d_model: int = 256
    dropout: float = 0.05

    def reset(self):
        """Reset all attributes to their default values."""
        self.num_layers = 0
        self.d_model = 0
        self.dropout = 0

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
        Generator to yield key-value pairs from a NetworkConfig instance.
        """
        for key, value in self.__dict__.items():
            yield key, value
    
class CooperativeNetworkConfig(NetworkConfig):
    def __init__(self, network_config: NetworkConfig, network_role_name: str, input_shape: Any, output_shape: Any, act_fn: str):
        super().__init__()
        self.apply_config(network_config.config_generator())  # Call the generator

        self.network_role_name = network_role_name
        self.input_shape = input_shape if isinstance(input_shape, (list, torch.Size, tuple)) else [input_shape]
        self.output_shape = output_shape if isinstance(output_shape, (list, torch.Size, tuple)) else [output_shape]
        self.act_fn = act_fn

@dataclass
class CCNetConfig(NetworkConfig):
    """
    Comprehensive configuration defining ccnet model settings.
    
    Attributes:
        model_name (str): Identifier for the core model.
        num_layers (int): Number of layers in the network.
        d_model (int): Dimensionality of the model's hidden layers.
        dropout (float): Dropout rate for regularization.
        obs_shape (list): Observational shape of the input.
        y_dim (int or None): Dimension parameter y.
        e_dim (int or None): Dimension parameter e.
        y_scale (int or None): Scaling factor for dimension y.
        task_type (str or None): Type of task the model is addressing.
        device (torch.device or None): Device to run the model on.
        use_seq_input (bool): Flag to indicate if sequential input is used.
    """
    model_name: str = None
    num_layers: int = 5
    d_model: int = 256
    dropout: float = 0.05
    obs_shape: List[Any] = field(default_factory=list)
    y_dim: int = None
    e_dim: int = None
    y_scale: int = None
    task_type: str = None
    device: torch.device = None
    use_seq_input: bool = False
    
    def __repr__(self):
        return (f"CCNetConfig(model_name='{self.model_name}', num_layers={self.num_layers}, "
                f"d_model={self.d_model}, dropout={self.dropout}, obs_shape={self.obs_shape}, "
                f"y_dim={self.y_dim}, e_dim={self.e_dim}, y_scale={self.y_scale}, "
                f"task_type={self.task_type}, device={self.device}, use_seq_input={self.use_seq_input})\n")