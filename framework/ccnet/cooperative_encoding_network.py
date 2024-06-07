'''
    Author:
        PARK, JunHo, junho@ccnets.org
'''
import torch
from .roles.explainer import Explainer
from .roles.reasoner import Reasoner
from .roles.producer import Producer

class CooperativeEncodingNetwork:
    def __init__(self, model_networks, network_params, algorithm_params, data_config, device):
        # Initialize model names and configurations.
        reset_pretrained = algorithm_params.reset_pretrained
        self.explainer = Explainer(model_networks[0], network_params, reset_pretrained, act_fn="layer_norm").to(device)
        self.reasoner = Reasoner(model_networks[1], network_params, reset_pretrained, act_fn="tanh").to(device)
        self.producer = Producer(model_networks[2], network_params, reset_pretrained, act_fn="none").to(device)

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
        self.task_type = data_config.task_type 
        
    def __set_train(self, train: bool):
        for network in self.networks:
            network.zero_grad()
            if train:
                network.train()
            else:
                network.eval()
                
    def _encode(self, input_data: torch.Tensor, padding_mask=None, batch_size=256) -> torch.Tensor:
        """
        Encodes input data into a tensor suitable for prediction tasks, using the explainer and 
        reasoner models. The output tensor contains all necessary variables for target prediction.

        Parameters:
        - input_data: A tensor representing the input data.

        Returns:
        - output_tensor: A tensor containing all variables necessary for target prediction.
        """
        input_data = input_data.to(self.device)
        batch_size = min(batch_size, input_data.size(0))
        encoded_tensor = torch.zeros(input_data.size(0), self.det_size).to(self.device)
        # encoded_tensor = torch.zeros(input_data.size(0), self.det_size + self.stoch_size).to(self.device)
        for i in range(0, input_data.size(0), batch_size):
            batch_input = input_data[i:i+batch_size]
            deterministic_variables = self.explainer(batch_input, padding_mask)
            # stochastic_variables = self.reasoner(batch_input, deterministic_variables, padding_mask)
            encoded_tensor[i:i+batch_size] = deterministic_variables
        return encoded_tensor
    
    def _decode(self, stochastic_variables: torch.Tensor, deterministic_variables: torch.Tensor, padding_mask = None, batch_size=256) -> torch.Tensor:
        """
        Decodes the encoded tensor using the producer model to reconstruct the input data.

        Parameters:
        - encoded_data: A tensor containing concatenated deterministic and stochastic variables.

        Returns:
        - reconstructed_data: A tensor representing the reconstructed data.
        """
        batch_size = min(batch_size, stochastic_variables.size(0))
        reconstructed_data = torch.zeros(stochastic_variables.size(0), *self.obs_shape).to(self.device)
        for i in range(0, stochastic_variables.size(0), batch_size):
            batch_stochastic = stochastic_variables[i:i+batch_size]
            batch_deterministic = deterministic_variables[i:i+batch_size]
            reconstructed_data[i:i+batch_size] = self.producer(batch_stochastic, batch_deterministic, padding_mask)
        return reconstructed_data
    
    def _decompose(self, input_data: torch.Tensor, padding_mask = None, batch_size=256) -> torch.Tensor:
        """
        Decomposes the input data into deterministic and stochastic variables using the explainer and reasoner models.
        
        Parameters:
        - input_data: A tensor representing the input data.
        - padding_mask: An optional mask tensor for padding variable lengths (default: None).
        
        Returns:
        - stochastic_variables: A tensor representing the stochastic variables.
        - deterministic_variables: A tensor representing the deterministic variables.
        """
        input_data = input_data.to(self.device)
        batch_size = min(batch_size, input_data.size(0))
        stochastic_variables = torch.zeros(input_data.size(0), self.stoch_size).to(self.device)
        deterministic_variables = torch.zeros(input_data.size(0), self.det_size).to(self.device)
        for i in range(0, input_data.size(0), batch_size):
            batch_input = input_data[i:i+batch_size]
            deterministic_variables[i:i+batch_size] = self.explainer(batch_input, padding_mask)
            stochastic_variables[i:i+batch_size] = self.reasoner(batch_input, deterministic_variables[i:i+batch_size], padding_mask)
        return stochastic_variables, deterministic_variables
        
    def encode(self, input_data: torch.Tensor, padding_mask=None, batch_size = 256) -> torch.Tensor:
        """
        Encodes input data into a tensor suitable for prediction tasks, using the explainer and 
        reasoner models. The output tensor contains all necessary variables for target prediction.

        Parameters:
        - input_data: A tensor representing the input data.

        Returns:
        - output_tensor: A tensor containing all variables necessary for target prediction.
        """
        input_data = input_data.to(self.device)
        
        with torch.no_grad():
            self.__set_train(False)
            encoded_tensor = self._encode(input_data, padding_mask, batch_size = batch_size)
            self.__set_train(True)
        return encoded_tensor

    def decode(self, stochastic_variables: torch.Tensor, deterministic_variables: torch.Tensor, padding_mask = None, batch_size = 256) -> torch.Tensor:
        """
        Decodes the encoded tensor using the producer model to reconstruct the input data.

        Parameters:
        - encoded_data: A tensor containing concatenated deterministic and stochastic variables.

        Returns:
        - reconstructed_data: A tensor representing the reconstructed data.
        """
        with torch.no_grad():
            self.__set_train(False)
            reconstructed_data = self._decode(stochastic_variables, deterministic_variables, padding_mask, batch_size = batch_size)
            self.__set_train(True)
        return reconstructed_data

    def decompose(self, input_data: torch.Tensor, padding_mask = None, batch_size = 256) -> torch.Tensor:
        """
        Decomposes the input data into deterministic and stochastic variables using the explainer and reasoner models.

        Parameters:
        - input_data: A tensor representing the input data.
        - padding_mask: An optional mask tensor for padding variable lengths (default: None).

        Returns:
        - stochastic_variables: A tensor representing the stochastic variables.
        - deterministic_variables: A tensor representing the deterministic variables.
        """
        with torch.no_grad():
            self.__set_train(False)
            stochastic_variables, deterministic_variables = self._decompose(input_data, padding_mask, batch_size = batch_size)
            self.__set_train(True)
        return stochastic_variables, deterministic_variables
        
    def synthesize(self, input_data: torch.Tensor, padding_mask=None, output_multiplier: int = None, batch_size = 256) -> torch.Tensor:
        """
        Synthesizes new data by cross-matching stochastic and deterministic variables from the encoded data,
        with options to either repeat or interleave these matches.

        Parameters:
        - input_data: A tensor containing the input data to be encoded.
        - padding_mask: An optional mask tensor for padding variable lengths (default: None).
        - output_multiplier: An integer indicating how many times the output dataset should be expanded relative to the input dataset.
        - interleave: A boolean indicating whether to interleave (True) or repeat (False) the combinations of stochastic and deterministic variables.

        Returns:
        - synthetic_data: A tensor representing the synthesized data.
        """
        with torch.no_grad():
            self.__set_train(False)
            stochastic_variables, deterministic_variables = self.decompose(input_data, padding_mask, batch_size = batch_size)
            
            batch_size = input_data.size(0)
            if output_multiplier is None:
                output_multiplier = batch_size
            else:
                output_multiplier = min(batch_size, output_multiplier)

            # Interleave the expanded data: s1, s2, s3, s1, s2, s3 and d1, d1, d1, d2, d2, d2
            indices = torch.arange(stochastic_variables.size(0)).repeat(output_multiplier)
            stochastic_expanded = stochastic_variables[indices % stochastic_variables.size(0)]
            deterministic_expanded = deterministic_variables.repeat(output_multiplier, 1)
            
            synthetic_data = self._decode(stochastic_expanded, deterministic_expanded, padding_mask, batch_size = batch_size)
            
            self.__set_train(True)
        
        return synthetic_data


