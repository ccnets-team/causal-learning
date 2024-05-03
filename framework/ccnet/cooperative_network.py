'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho@ccnets.org
'''
import torch
from .roles.explainer import Explainer
from .roles.reasoner import Reasoner
from .roles.producer import Producer
from tools.tensor import adjust_tensor_dim
                    
class CooperativeNetwork:
    def __init__(self, model_params, task_type, state_size, label_size, explain_size, device, 
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
        explainer_params = model_params.core_params
        reasoner_params = model_params.core_params
        producer_params = model_params.core_params
        
        explainer_network = model_params.core_networks[0]
        reasoner_network = model_params.core_networks[1]
        producer_network = model_params.core_networks[2]

        self.encoder = encoder
        if task_type == "classification":
            reasoner_act_fn = 'softmax'
        elif task_type == "regression" or task_type == "binary":
            reasoner_act_fn = 'sigmoid'
        else:
            raise ValueError(f"Invalid task type: {task_type}")
            
        # Create and initialize each component model and move to the specified device.
        self.network_names = ["explainer", "reasoner", "producer"]
        self.explainer =  Explainer(explainer_network, explainer_params, [state_size], explain_size, act_fn="layer_norm").to(device)
        self.reasoner =  Reasoner(reasoner_network, reasoner_params, [state_size], explain_size, label_size, act_fn=reasoner_act_fn).to(device)
        self.producer =  Producer(producer_network, producer_params, label_size, explain_size, [state_size], act_fn="none").to(device)
        self.networks = [self.explainer, self.reasoner, self.producer]
        
        self.explain_size = explain_size
        self.state_size = state_size
        self.label_size = label_size
        self.__device = device

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
            original_dim = len(encoded_input.shape)
            encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            explanation = self.explainer(encoded_input)
            reasoned_output = self.reasoner(encoded_input, explanation)
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
            original_dim = len(encoded_input.shape)
            encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            reasoned_output = self.reasoner(encoded_input, explanation)
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
            original_dim = len(condition_data.shape)
            condition_data = adjust_tensor_dim(condition_data, target_dim=3)
            random_explanation = torch.randn(condition_data.size(0), self.explanation_size).to(self.device)   
            generated_output = self.producer(condition_data, random_explanation)
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
            original_dim = len(condition_data.shape)
            condition_data = adjust_tensor_dim(condition_data, target_dim=3)
            explanation = adjust_tensor_dim(explanation, target_dim=3)
            produced_output = self.producer(condition_data, explanation)
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
            original_dim = len(encoded_input.shape)
            encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            explanation = self.explainer(encoded_input)
            inferred_output = self.reasoner(encoded_input, explanation)
            reconstructed_output = self.producer(inferred_output, explanation)
            reconstructed_output = adjust_tensor_dim(reconstructed_output, target_dim=original_dim)
            reconstructed_data = self.decode(reconstructed_output)
        return reconstructed_data