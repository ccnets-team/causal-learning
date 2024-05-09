'''
    Author:
        PARK, JunHo, junho@ccnets.org
'''
import torch
from .roles.explainer import Explainer
from .roles.reasoner import Reasoner
from .roles.producer import Producer

class CooperativeEncodingNetwork:
    def __init__(self, model_networks, network_params, device):
        # Initialize model names and configurations.
        self.explainer = Explainer(model_networks[0], network_params, act_fn="layer_norm").to(device)
        self.reasoner = Reasoner(model_networks[1], network_params, act_fn="layer_norm").to(device)
        self.producer = Producer(model_networks[2], network_params, act_fn="none").to(device)

        model_name = network_params.model_name
        # Add model_name prefix to the network names
        network_names = ["explainer", "reasoner", "producer"]
        self.model_name = model_name
        self.network_names = [f"{model_name}_{name}" for name in network_names]
        self.networks = [self.explainer, self.reasoner, self.producer]

        self.obs_shape = network_params.obs_shape
        self.det_size = network_params.z_dim
        self.stoch_size = network_params.condition_dim
        self.device = device
        
    def __set_train(self, train: bool):
        for network in self.networks:
            network.zero_grad()
            if train:
                network.train()
            else:
                network.eval()
                            
    def encode(self, input_data: torch.Tensor, padding_mask = None) -> torch.Tensor:
        """
        Encodes input data using the explainer and reasoner models.

        Parameters:
        - input_data: A tensor representing the input data.

        Returns:
        - encoded_data: A concatenated tensor of encoded deterministic and stochastic variables.
        """
        with torch.no_grad():
            self.__set_train(False)
            deterministic_variables = self.explainer(input_data, padding_mask)
            stochastic_variables = self.reasoner(input_data, deterministic_variables, padding_mask)
            encoded_data = torch.cat([stochastic_variables, deterministic_variables], dim=-1)
            self.__set_train(True)
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
            self.__set_train(False)
            stochastic_variables = encoded_data[..., :self.stoch_size]
            deterministic_variables = encoded_data[..., self.stoch_size:]
            reconstructed_data = self.producer(stochastic_variables, deterministic_variables, padding_mask)
            self.__set_train(True)
        return reconstructed_data
    
    def synthesize(self, input_data: torch.Tensor, padding_mask=None, output_multiplier: int = 10) -> torch.Tensor:
        """
        Synthesizes new data by cross-matching stochastic and deterministic variables from the encoded data,
        with control over the expansion of the output dataset relative to the input dataset.

        This method encodes input data into stochastic and deterministic parts, cross matches them
        to explore possible combinations, and uses the producer model to generate new synthetic data based on these combinations.
        The number of output data points is a multiple of the input data points, controlled by the output_multiplier.

        Parameters:
        - input_data: A tensor containing the input data to be encoded.
        - padding_mask: An optional mask tensor for padding variable lengths (default: None).
        - output_multiplier: An integer indicating how many times the output dataset should be larger than the input dataset (default: 10).

        Returns:
        - synthetic_data: A tensor representing the synthesized data.
        """
        with torch.no_grad():
            self.__set_train(False)
            encoded_data = self.encode(input_data, padding_mask)
            stochastic_variables = encoded_data[..., :self.stoch_size]
            deterministic_variables = encoded_data[..., self.stoch_size:]

            # Create all possible combinations of stochastic and deterministic variables:
            num_examples = stochastic_variables.size(0)
            stochastic_expanded = stochastic_variables.unsqueeze(1).expand(-1, num_examples, -1)
            deterministic_expanded = deterministic_variables.unsqueeze(0).expand(num_examples, -1, -1)
            combined_data = torch.cat((stochastic_expanded, deterministic_expanded), dim=-1)

            # Flatten the combinations and shuffle:
            combined_data = combined_data.view(-1, combined_data.size(-1))
            shuffled_indices = torch.randperm(combined_data.size(0))

            # Calculate the desired number of outputs:
            desired_output_count = min(len(shuffled_indices), num_examples * output_multiplier)
            selected_indices = shuffled_indices[:desired_output_count]
            selected_combinations = combined_data[selected_indices]

            synthetic_data = self.producer(selected_combinations, padding_mask)
            
            self.__set_train(True)
        
        return synthetic_data

