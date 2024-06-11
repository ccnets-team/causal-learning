'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho@ccnets.org
'''
import torch
from .roles.explainer import Explainer
from .roles.reasoner import Reasoner
from .roles.producer import Producer
from tools.tensor_utils import adjust_tensor_dim
from framework.utils.ccnet_utils import determine_activation_by_task_type, generate_condition_data

class CooperativeNetwork:
    def __init__(self, model_networks, network_params, algorithm_params, data_config, device, encoder = None):
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
        if network_params.model_name == 'gpt':
            self.use_seq = True 
        else:
            self.use_seq = False
        self.task_type, self.label_size, self.label_scale = ccnet.task_type, ccnet.label_size, ccnet.label_scale
        task_act_fn = determine_activation_by_task_type(self.task_type)
            
        model_name = network_params.model_name
        # Add model_name prefix to the network names
        network_names = ["explainer", "reasoner", "producer"]
        self.model_name = model_name
        self.network_names = [f"{model_name}_{name}" for name in network_names]
        reset_pretrained = algorithm_params.reset_pretrained
        self.explainer =  Explainer(model_networks[0], network_params, reset_pretrained, act_fn=data_config.explain_layer).to(device)
        self.reasoner =  Reasoner(model_networks[1], network_params, reset_pretrained, act_fn=task_act_fn).to(device)
        self.producer =  Producer(model_networks[2], network_params, reset_pretrained, act_fn="none").to(device)
        self.networks = [self.explainer, self.reasoner, self.producer]
        
        self.explain_size = network_params.z_dim
        self.label_size = network_params.condition_dim
        self.device = device
        self.task_act_fn = task_act_fn
        self.encoder = encoder
    
    def __set_train(self, train: bool):
        for network in self.networks:
            network.zero_grad()
            if train:
                network.train()
            else:
                network.eval()

    def encode(self, data, padding_mask = None, batch_size = 256):
        """
        Encodes the input data using the encoder if available.

        Args:
            data (Tensor): Input data tensor.

        Returns:
            Tensor: Encoded data.
        """
        if self.encoder:
            data = self.encoder.encode(data, padding_mask, batch_size = batch_size)
        return data

    def decode(self, encoded_data, padding_mask = None, batch_size = 256):
        """
        Decodes the input data using the decoder if available.

        Args:
            encoded_data (Tensor): Encoded data tensor.

        Returns:
            Tensor: Decoded data.
        """
        if self.encoder:
            encoded_data = self.encoder.decode(encoded_data, padding_mask, batch_size = batch_size)
        return encoded_data
    
    def _explain(self, input_data, padding_mask=None, batch_size=256):
        """
        Generates an explanation for the input data by processing it in batches. This method is useful
        for handling large datasets that may not fit into memory if processed all at once. It iteratively
        processes subsets of the input data, generating explanations for each batch and then concatenating
        these to form the final output.

        Args:
            input_data (Tensor): Input data tensor. This is the data for which explanations are generated.
            padding_mask (Tensor, optional): An optional mask for the input data that indicates which elements
                                            should be ignored during the explanation generation process.
            batch_size (int): The number of examples to process in each batch. This controls the maximum number
                            of examples that are processed at a time, which helps manage memory usage.

        Returns:
            Tensor: A tensor containing explanations for each example in the input data, concatenated along the
                    first dimension.
        """
        # Initialize a list to hold batched results
        batched_explanations = []

        # Process input data in batches
        for i in range(0, input_data.size(0), batch_size):
            batch_input = input_data[i:i + batch_size]
            batch_mask = padding_mask[i:i + batch_size] if padding_mask is not None else None

            # Generate explanation for the batch
            batch_explanation = self.explainer(batch_input, batch_mask)
            batched_explanations.append(batch_explanation)

        # Concatenate all batched explanations along the first dimension
        explanations = torch.cat(batched_explanations, dim=0)
        return explanations
    
    def _reason(self, input_data, explanation, padding_mask=None, batch_size=256):
        """
        Uses the explanations and input data to reason about the output by processing them in batches. This method
        is useful for handling large datasets that may not fit into memory if processed all at once. It iteratively
        processes subsets of the input data and explanations, reasoning about the output for each batch and then
        concatenating these to form the final output.

        Args:
            input_data (Tensor): Input data tensor. This is the data for which output is reasoned about.
            explanation (Tensor): Explanation tensor. This is the explanation for the input data.
            padding_mask (Tensor, optional): An optional mask for the input data that indicates which elements
                                            should be ignored during the reasoning process.
            batch_size (int): The number of examples to process in each batch. This controls the maximum number
                            of examples that are processed at a time, which helps manage memory usage.

        Returns:
            Tensor: A tensor containing reasoned output for each example in the input data, concatenated along the
                    first dimension.
        """
        # Initialize a list to hold batched results
        batched_outputs = []

        # Process input data in batches
        for i in range(0, input_data.size(0), batch_size):
            batch_input = input_data[i:i + batch_size]
            batch_explanation = explanation[i:i + batch_size]
            batch_mask = padding_mask[i:i + batch_size] if padding_mask is not None else None

            # Reason about the output for the batch
            batch_output = self.reasoner(batch_input, batch_explanation, batch_mask)
            batched_outputs.append(batch_output)

        # Concatenate all batched outputs along the first dimension
        outputs = torch.cat(batched_outputs, dim=0)
        return outputs

    def _produce(self, condition_data, explanation, padding_mask = None, batch_size=256):
        """
        Generates new data based on conditions and explanations by processing them in batches. This method is useful
        for handling large datasets that may not fit into memory if processed all at once. It iteratively processes
        subsets of the condition data and explanations, generating new data for each batch and then concatenating
        these to form the final output.

        Args:
            condition_data (Tensor): Condition data tensor. This is the data that conditions the generation process.
            explanation (Tensor): Explanation tensor. This is the explanation for the condition data.
            padding_mask (Tensor, optional): An optional mask for the condition data that indicates which elements
                                            should be ignored during the generation process.
            batch_size (int): The number of examples to process in each batch. This controls the maximum number
                            of examples that are processed at a time, which helps manage memory usage.

        Returns:
            Tensor: A tensor containing generated output data for each example in the condition data, concatenated
                    along the first dimension.
        """
        # Initialize a list to hold batched results
        batched_outputs = []

        # Process condition data in batches
        for i in range(0, condition_data.size(0), batch_size):
            batch_condition = condition_data[i:i + batch_size]
            batch_explanation = explanation[i:i + batch_size]
            batch_mask = padding_mask[i:i + batch_size] if padding_mask is not None else None

            # Generate new data for the batch
            batch_output = self.producer(batch_condition, batch_explanation, batch_mask)
            batched_outputs.append(batch_output)

        # Concatenate all batched outputs along the first dimension
        outputs = torch.cat(batched_outputs, dim=0)
        return outputs          
    
    def explain(self, input_data, padding_mask = None, batch_size=256):
        """
        Generates an explanation for the input data without updating the explainer model.

        Args:
            input_data (Tensor): Input data tensor.

        Returns:
            Tensor: Explanation tensor.
        """
        with torch.no_grad():
            self.__set_train(False)
            encoded_input = self.encode(input_data, padding_mask, batch_size = batch_size)
            if self.use_seq:
                encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            explanation = self._explain(encoded_input, padding_mask, batch_size = batch_size)
            self.__set_train(True)
        return explanation

    def infer(self, input_data, padding_mask = None, batch_size=256):
        """
        Infers output from input data using the explainer and reasoner models without updating them.

        Args:
            input_data (Tensor): Input data tensor.

        Returns:
            Tensor: Inferred output tensor.
        """
        with torch.no_grad():
            self.__set_train(False)
            encoded_input = self.encode(input_data, padding_mask, batch_size = batch_size)
            if self.use_seq:
                original_dim = len(encoded_input.shape)
                encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            explanation = self._explain(encoded_input, padding_mask, batch_size = batch_size)
            reasoned_output = self._reason(encoded_input, explanation, padding_mask, batch_size = batch_size)
            if self.use_seq:
                reasoned_output = adjust_tensor_dim(reasoned_output, target_dim=original_dim)
            self.__set_train(True)
        return reasoned_output
    
    def reason(self, input_data, explanation, padding_mask = None, batch_size=256):
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
            encoded_input = self.encode(input_data, padding_mask, batch_size = batch_size)
            if self.use_seq:
                original_dim = len(encoded_input.shape)
                encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            reasoned_output = self._reason(encoded_input, explanation, padding_mask, batch_size = batch_size)
            if self.use_seq:
                reasoned_output = adjust_tensor_dim(reasoned_output, target_dim=original_dim)
            self.__set_train(True)
        return reasoned_output
            
    def generate(self, explanation, padding_mask=None, batch_size=256):
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
            generated_output = self._produce(condition_data, explanation, padding_mask, batch_size = batch_size)
            generated_data = self.decode(generated_output, padding_mask, batch_size = batch_size)
            self.__set_train(True)

        return generated_data, condition_data

    def produce(self, condition_data, explanation, padding_mask = None, batch_size = 256):
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
            if self.use_seq:
                original_dim = len(condition_data.shape)
                condition_data = adjust_tensor_dim(condition_data, target_dim=3)
                explanation = adjust_tensor_dim(explanation, target_dim=3)
            produced_output = self._produce(condition_data, explanation, padding_mask, batch_size = batch_size)
            if self.use_seq:
                produced_output = adjust_tensor_dim(produced_output, target_dim=original_dim)
            produced_data = self.decode(produced_output, padding_mask, batch_size = batch_size)
            self.__set_train(True)
        return produced_data
    
    def reconstruct(self, input_data, padding_mask = None, batch_size = 256):
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
            encoded_input = self.encode(input_data, padding_mask, batch_size = batch_size)
            if self.use_seq:
                original_dim = len(encoded_input.shape)
                encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
            explanation = self._explain(encoded_input, padding_mask, batch_size = batch_size)
            inferred_output = self._reason(encoded_input, explanation, padding_mask, batch_size = batch_size)
            reconstructed_output = self._produce(inferred_output, explanation, padding_mask, batch_size = batch_size)
            if self.use_seq:
                reconstructed_output = adjust_tensor_dim(reconstructed_output, target_dim=original_dim)
            reconstructed_data = self.decode(reconstructed_output, padding_mask, batch_size = batch_size)
            self.__set_train(True)
        return reconstructed_data
    
    def counter_generate(self, input_data, condition_data, padding_mask=None, batch_size=256):
        """
        Generates a new version of the input data by integrating counterfactual conditions. This function performs a
        three-step process: encoding the input, explaining to generate an explanation vector, and then producing
        an output conditioned on the counterfactual data. All steps are performed without updating the underlying model weights.

        Args:
            input_data (Tensor): The original input data tensor that you want to generate counterfactuals for.
            condition_data (Tensor): The counterfactual condition data tensor that specifies the desired state or conditions
                                    for generating the counterfactual outcomes.
            padding_mask (Optional[Tensor]): An optional mask tensor to ignore certain parts of the input data during processing.
                                            Default is None.
            batch_size (int): The number of samples to process in a single batch. Default is 256.

        Returns:
            Tensor: The tensor containing the generated data based on the counterfactual conditions, which hypothetically
                    represents how the input data might appear under different specified conditions.
        """
        with torch.no_grad():
            self.__set_train(False)
            encoded_input = self.encode(input_data, padding_mask, batch_size=batch_size)
            if self.use_seq:
                original_dim = len(encoded_input.shape)
                encoded_input = adjust_tensor_dim(encoded_input, target_dim=3)
                condition_data = adjust_tensor_dim(condition_data, target_dim=3)
            explanation = self._explain(encoded_input, padding_mask, batch_size=batch_size)
            counter_output = self._produce(condition_data, explanation, padding_mask, batch_size=batch_size)
            if self.use_seq:
                counter_output = adjust_tensor_dim(counter_output, target_dim=original_dim)
            counter_generated_data = self.decode(counter_output, padding_mask, batch_size=batch_size)
            self.__set_train(True)
        return counter_generated_data

     