'''
    Author:
        PARK, JunHo, junho@ccnets.org
'''
import torch
from .roles.explainer import Explainer
from .roles.reasoner import Reasoner
from .roles.producer import Producer

class CooperativeEncodingNetwork:
    def __init__(self, model_params, obs_shape, stoch_size, det_size, device):
        # Initialize model names and configurations.
        encoding_networks = model_params.encoding_networks
        encoding_params = model_params.encoding_params
        encoding_params.obs_shape = obs_shape
        encoding_params.z_dim = det_size
        encoding_params.condition_dim = stoch_size
            
        self.explainer = Explainer(encoding_networks[0], encoding_params, obs_shape, det_size, act_fn="layer_norm").to(device)
        self.reasoner = Reasoner(encoding_networks[1], encoding_params, obs_shape, det_size, stoch_size, act_fn="layer_norm").to(device)
        self.producer = Producer(encoding_networks[2], encoding_params, stoch_size, det_size, obs_shape, act_fn="none").to(device)

        self.network_names = ["explainer", "reasoner", "producer"]
        self.networks = [self.explainer, self.reasoner, self.producer]
        
        self.obs_shape = obs_shape
        self.det_size = det_size
        self.stoch_size = stoch_size
            
    def encode(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Encodes input data using the explainer and reasoner models.

        Parameters:
        - input_data: A tensor representing the input data.

        Returns:
        - encoded_data: A concatenated tensor of encoded deterministic and stochastic variables.
        """
        with torch.no_grad():
            deterministic_variables = self.explainer(input_data)
            stochastic_variables = self.reasoner(input_data, deterministic_variables)
            encoded_data = torch.cat([stochastic_variables, deterministic_variables], dim=-1)
        return encoded_data

    def decode(self, encoded_data: torch.Tensor) -> torch.Tensor:
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
            reconstructed_data = self.producer(stochastic_variables, deterministic_variables)
        return reconstructed_data