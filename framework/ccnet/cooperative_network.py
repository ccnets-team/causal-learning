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
from tools.tensor_utils import adjust_tensor_dim
from framework.utils.ccnet_utils import determine_activation_by_task_type, generate_condition_data
import torch.nn.functional as F

class CooperativeNetwork:
    def __init__(self, model_networks, network_params, task_type, device, encoder = None):
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
        if isinstance(network_params, GPTModelParams):
            self.use_gpt = True 
        elif isinstance(network_params, ImageModelParams):
            self.use_gpt = False

        task_act_fn = determine_activation_by_task_type(task_type)
            
        model_name = network_params.model_name
        # Add model_name prefix to the network names
        network_names = ["explainer", "reasoner", "producer"]
        self.model_name = model_name
        self.network_names = [f"{model_name}_{name}" for name in network_names]
        self.explainer =  Explainer(model_networks[0], network_params, act_fn="layer_norm").to(device)
        self.reasoner =  Reasoner(model_networks[1], network_params, act_fn=task_act_fn).to(device)
        self.producer =  Producer(model_networks[2], network_params, act_fn="none").to(device)
        self.networks = [self.explainer, self.reasoner, self.producer]
        
        self.explain_size = network_params.z_dim
        self.label_size = network_params.condition_dim
        self.device = device
        self.task_act_fn = task_act_fn
        self.task_type = task_type
        self.encoder = encoder
    
    def __set_train(self, train: bool):
        for network in self.networks:
            network.zero_grad()
            if train:
                network.train()
            else:
                network.eval()

    def encode(self, data, padding_mask = None):
        """
        Encodes the input data using the encoder if available.

        Args:
            data (Tensor): Input data tensor.

        Returns:
            Tensor: Encoded data.
        """
        if self.encoder:
            data = self.encoder.encode(data, padding_mask)
        return data

    def decode(self, encoded_data, padding_mask = None):
        """
        Decodes the input data using the decoder if available.

        Args:
            encoded_data (Tensor): Encoded data tensor.

        Returns:
            Tensor: Decoded data.
        """
        if self.encoder:
            encoded_data = self.encoder.decode(encoded_data, padding_mask)
        return encoded_data
   
    def explain(self, input_data, padding_mask = None):
        """
        Generates an explanation for the input data without updating the explainer model.

        Args:
            input_data (Tensor): Input data tensor.

        Returns:
            Tensor: Explanation tensor.
        """
        with torch.no_grad():
            self.__set_train(False)
            encoded_input = self.encode(input_data, padding_mask)
            if self.use_gpt:
                encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            explanation = self.explainer(encoded_input, padding_mask)
            self.__set_train(True)
        return explanation

    def infer(self, input_data, padding_mask = None):
        """
        Infers output from input data using the explainer and reasoner models without updating them.

        Args:
            input_data (Tensor): Input data tensor.

        Returns:
            Tensor: Inferred output tensor.
        """
        with torch.no_grad():
            self.__set_train(False)
            encoded_input = self.encode(input_data, padding_mask)
            if self.use_gpt:
                original_dim = len(encoded_input.shape)
                encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            explanation = self.explainer(encoded_input, padding_mask)
            reasoned_output = self.reasoner(encoded_input, explanation, padding_mask)
            if self.use_gpt:
                reasoned_output = adjust_tensor_dim(reasoned_output, target_dim=original_dim)
            self.__set_train(True)
        return reasoned_output
    
    def reason(self, input_data, explanation, padding_mask = None):
        """
        Uses the explanations and input data to reason about the output without updating the model.

        Args:
            input_data (Tensor): Input data tensor.
            explanation (Tensor): Explanation tensor.

        Returns:
            Tensor: Reasoned output tensor.
        """
        with torch.no_grad():
            self.__set_train(False)
            encoded_input = self.encode(input_data, padding_mask)
            if self.use_gpt:
                original_dim = len(encoded_input.shape)
                encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            reasoned_output = self.reasoner(encoded_input, explanation, padding_mask)
            if self.use_gpt:
                reasoned_output = adjust_tensor_dim(reasoned_output, target_dim=original_dim)
            self.__set_train(True)
        return reasoned_output

    def generate(self, explanation, padding_mask=None):
        """
        Generates new data based on input explanations with random discrete conditions without updating the producer model.

        Args:
            explanation (Tensor): Explanation tensor.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the generated output data tensor and the condition data tensor.
        """
        with torch.no_grad():
            self.__set_train(False)
            label_shape = explanation.shape[:-1] + (self.label_size,)
            condition_data = generate_condition_data(label_shape, self.task_type, self.device)
            generated_output = self.producer(condition_data, explanation, padding_mask)
            generated_data = self.decode(generated_output, padding_mask)
            self.__set_train(True)

        return generated_data, condition_data

    def produce(self, condition_data, explanation, padding_mask = None):
        """
        Generates new data based on conditions and explanations without updating the producer model.

        Args:
            condition_data (Tensor): Condition data tensor.
            explanation (Tensor): Explanation tensor.

        Returns:
            Tensor: Produced output data tensor.
        """
        with torch.no_grad():
            self.__set_train(False)
            if self.use_gpt:
                original_dim = len(condition_data.shape)
                condition_data = adjust_tensor_dim(condition_data, target_dim=3)
                explanation = adjust_tensor_dim(explanation, target_dim=3)
            produced_output = self.producer(condition_data, explanation, padding_mask)
            if self.use_gpt:
                produced_output = adjust_tensor_dim(produced_output, target_dim=original_dim)
            produced_data = self.decode(produced_output, padding_mask)
            self.__set_train(True)
        return produced_data
    
    def reconstruct(self, input_data, padding_mask = None):
        """
        Reconstructs the input data by first explaining, then reasoning, and finally producing the output,
        all without updating the models.

        Args:
            input_data (Tensor): Input data tensor.

        Returns:
            Tensor: Reconstructed output data tensor.
        """
        with torch.no_grad():
            self.__set_train(False)
            encoded_input = self.encode(input_data, padding_mask)
            if self.use_gpt:
                original_dim = len(encoded_input.shape)
                encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            explanation = self.explainer(encoded_input, padding_mask)
            inferred_output = self.reasoner(encoded_input, explanation, padding_mask)
            reconstructed_output = self.producer(inferred_output, explanation, padding_mask)
            if self.use_gpt:
                reconstructed_output = adjust_tensor_dim(reconstructed_output, target_dim=original_dim)
            reconstructed_data = self.decode(reconstructed_output, padding_mask)
            self.__set_train(True)
        return reconstructed_data