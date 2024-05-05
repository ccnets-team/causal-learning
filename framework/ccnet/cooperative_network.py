'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho@ccnets.org
'''
import torch
from .roles.explainer import Explainer
from .roles.reasoner import Reasoner
from .roles.producer import Producer
from tools.setting.ml_params import GPTModelParams, ImageModelParams 
from tools.tensor import adjust_tensor_dim, determine_activation_by_task_type
                    
class CooperativeNetwork:
    def __init__(self, model_name, model_params, task_type, obs_shape, label_size, explain_size, device, 
                 encoder = None):
        """
        Initializes the Cooperative Network with specified model parameters and computational device.

        Args:
            model_params (object): Contains individual network parameters and configurations.
            state_size (int): Size of the state input to the models.
            label_size (int): Size of the output labels.
            explain_size (int): Size of the explanations generated.
            device (str): Device ('cpu' or 'cuda') to run the models on.
            encoder (optional): Encoder object for input data preprocessing.
        """
        # Initialize model names and configurations.        
        core_params = model_params.core_params
        
        if isinstance(core_params, GPTModelParams):
            self.use_gpt = True 
        elif isinstance(core_params, ImageModelParams):
            self.use_gpt = False 
            core_params.obs_shape = obs_shape
            core_params.z_dim = explain_size
            core_params.condition_dim = label_size
        
        explainer_network = model_params.core_networks[0]
        reasoner_network = model_params.core_networks[1]
        producer_network = model_params.core_networks[2]

        self.encoder = encoder
        task_act_fn = determine_activation_by_task_type(task_type)
            
        # Add model_name prefix to the network names
        network_names = ["explainer", "reasoner", "producer"]
        self.model_name = model_name
        self.network_names = [f"{model_name}_{name}" for name in network_names]
        self.explainer =  Explainer(explainer_network, core_params, obs_shape, explain_size, act_fn="layer_norm").to(device)
        self.reasoner =  Reasoner(reasoner_network, core_params, obs_shape, explain_size, label_size, act_fn=task_act_fn).to(device)
        self.producer =  Producer(producer_network, core_params, label_size, explain_size, obs_shape, act_fn="none").to(device)
        self.networks = [self.explainer, self.reasoner, self.producer]
        
        self.explain_size = explain_size
        self.label_size = label_size
        self.device = device

    def encode(self, data):
        """
        Encodes the input data using the encoder if available.

        Args:
            data (Tensor): Input data tensor.

        Returns:
            Tensor: Encoded data.
        """
        if self.encoder:
            data = self.encoder.encode(data)
        return data

    def decode(self, encoded_data):
        """
        Decodes the input data using the decoder if available.

        Args:
            encoded_data (Tensor): Encoded data tensor.

        Returns:
            Tensor: Decoded data.
        """
        if self.encoder:
            encoded_data = self.encoder.decode(encoded_data)
        return encoded_data
   
    def explain(self, input_data):
        """
        Generates an explanation for the input data without updating the explainer model.

        Args:
            input_data (Tensor): Input data tensor.

        Returns:
            Tensor: Explanation tensor.
        """
        with torch.no_grad():
            encoded_input = self.encode(input_data)
            if self.use_gpt:
                encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            explanation = self.explainer(encoded_input)
        return explanation

    def infer(self, input_data):
        """
        Infers output from input data using the explainer and reasoner models without updating them.

        Args:
            input_data (Tensor): Input data tensor.

        Returns:
            Tensor: Inferred output tensor.
        """
        with torch.no_grad():
            encoded_input = self.encode(input_data)
            if self.use_gpt:
                original_dim = len(encoded_input.shape)
                encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            explanation = self.explainer(encoded_input)
            reasoned_output = self.reasoner(encoded_input, explanation)
            if self.use_gpt:
                reasoned_output = adjust_tensor_dim(reasoned_output, target_dim=original_dim)
        return reasoned_output
    
    def reason(self, input_data, explanation):
        """
        Uses the explanations and input data to reason about the output without updating the model.

        Args:
            input_data (Tensor): Input data tensor.
            explanation (Tensor): Explanation tensor.

        Returns:
            Tensor: Reasoned output tensor.
        """
        with torch.no_grad():
            encoded_input = self.encode(input_data)
            if self.use_gpt:
                original_dim = len(encoded_input.shape)
                encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            reasoned_output = self.reasoner(encoded_input, explanation)
            if self.use_gpt:
                reasoned_output = adjust_tensor_dim(reasoned_output, target_dim=original_dim)
        return reasoned_output

    def generate(self, condition_data):
        """
        Generates new data based on input conditions using random explanations without updating the producer model.

        Args:
            condition_data (Tensor): Input condition data tensor.

        Returns:
            Tensor: Generated output data tensor.
        """
        with torch.no_grad():
            if self.use_gpt:
                original_dim = len(condition_data.shape)
                condition_data = adjust_tensor_dim(condition_data, target_dim=3)
            random_explanation = torch.randn(condition_data.size(0), self.explanation_size).to(self.device)   
            generated_output = self.producer(condition_data, random_explanation)
            if self.use_gpt:
                generated_output = adjust_tensor_dim(generated_output, target_dim=original_dim)
            generated_data = self.decode(generated_output)
        return generated_data

    def produce(self, condition_data, explanation):
        """
        Generates new data based on conditions and explanations without updating the producer model.

        Args:
            condition_data (Tensor): Condition data tensor.
            explanation (Tensor): Explanation tensor.

        Returns:
            Tensor: Produced output data tensor.
        """
        with torch.no_grad():
            if self.use_gpt:
                original_dim = len(condition_data.shape)
                condition_data = adjust_tensor_dim(condition_data, target_dim=3)
                explanation = adjust_tensor_dim(explanation, target_dim=3)
            produced_output = self.producer(condition_data, explanation)
            if self.use_gpt:
                produced_output = adjust_tensor_dim(produced_output, target_dim=original_dim)
            produced_data = self.decode(produced_output)
        return produced_data
    
    def reconstruct(self, input_data):
        """
        Reconstructs the input data by first explaining, then reasoning, and finally producing the output,
        all without updating the models.

        Args:
            input_data (Tensor): Input data tensor.

        Returns:
            Tensor: Reconstructed output data tensor.
        """
        with torch.no_grad():
            encoded_input = self.encode(input_data)
            if self.use_gpt:
                original_dim = len(encoded_input.shape)
                encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            explanation = self.explainer(encoded_input)
            inferred_output = self.reasoner(encoded_input, explanation)
            reconstructed_output = self.producer(inferred_output, explanation)
            if self.use_gpt:
                reconstructed_output = adjust_tensor_dim(reconstructed_output, target_dim=original_dim)
            reconstructed_data = self.decode(reconstructed_output)
        return reconstructed_data