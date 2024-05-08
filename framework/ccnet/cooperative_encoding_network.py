'''
    Author:
        PARK, JunHo, junho@ccnets.org
'''
import torch
from .roles.explainer import Explainer
from .roles.reasoner import Reasoner
from .roles.producer import Producer
from tools.setting.ml_params import GPTModelParams, ImageModelParams 

class CooperativeEncodingNetwork:
    def __init__(self, model_name, networks, network_params, obs_shape, stoch_size, det_size, device):
        # Initialize model names and configurations.
        
        explainer_network = networks[0]
        reasoner_network = networks[1]
        producer_network = networks[2]

        if isinstance(network_params, ImageModelParams):
            network_params.obs_shape = obs_shape
            network_params.z_dim = det_size
            network_params.condition_dim = stoch_size
            
        self.explainer = Explainer(explainer_network, network_params, obs_shape, det_size, act_fn="layer_norm").to(device)
        self.reasoner = Reasoner(reasoner_network, network_params, obs_shape, det_size, stoch_size, act_fn="layer_norm").to(device)
        self.producer = Producer(producer_network, network_params, stoch_size, det_size, obs_shape, act_fn="none").to(device)

        # Add model_name prefix to the network names
        network_names = ["explainer", "reasoner", "producer"]
        self.model_name = model_name
        self.network_names = [f"{model_name}_{name}" for name in network_names]
        self.networks = [self.explainer, self.reasoner, self.producer]
        self.obs_shape = obs_shape
        self.det_size = det_size
        self.stoch_size = stoch_size
        self.device = device
            
    def encode(self, input_data: torch.Tensor, padding_mask = None) -> torch.Tensor:
        """
        Encodes input data using the explainer and reasoner models.

        Parameters:
        - input_data: A tensor representing the input data.

        Returns:
        - encoded_data: A concatenated tensor of encoded deterministic and stochastic variables.
        """
        with torch.no_grad():
            deterministic_variables = self.explainer(input_data, padding_mask)
            stochastic_variables = self.reasoner(input_data, deterministic_variables, padding_mask)
            encoded_data = torch.cat([stochastic_variables, deterministic_variables], dim=-1)
        return encoded_data

    def decode(self, encoded_data: torch.Tensor, padding_mask = None) -> torch.Tensor:
        """
        Decodes the encoded tensor using the producer model to reconstruct the input data.

        Parameters:
        - encoded_data: A tensor containing concatenated deterministic and stochastic variables.

        Returns:
        - reconstructed_data: A tensor representing the reconstructed data.
        """
        with torch.no_grad():
            stochastic_variables = encoded_data[..., :self.stoch_size]
            deterministic_variables = encoded_data[..., self.stoch_size:]
            reconstructed_data = self.producer(stochastic_variables, deterministic_variables, padding_mask)
        return reconstructed_data