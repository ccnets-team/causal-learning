'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho@ccnets.org
'''
import torch
from .roles.explainer import Explainer
from .roles.reasoner import Reasoner
from .roles.producer import Producer
from tools.tensor.batch import adjust_tensor_dim
from ccnet.utils import determine_activation_function, generate_condition_data

class CausalCooperativeNet:
    def __init__(self, networks, ccnet_config, device):
        """
        Initializes the Cooperative Network with specified model parameters and computational device.

        Args:
            ccnet_config (object): Contains individual network parameters and configurations.
            label_size (int): Size of the output labels.
            explain_size (int): Size of the explanations generated.
            device (str): Device ('cpu' or 'cuda') to run the models on.
        """
        # Initialize model names and configurations.        
        self.use_seq_input = ccnet_config.use_seq_input
        self.task_type, self.label_scale = ccnet_config.task_type, ccnet_config.y_scale
        task_act_fn = determine_activation_function(self.task_type)
            
        # Add model_name prefix to the network names
        network_names = ["explainer", "reasoner", "producer"]
        self.model_name = ccnet_config.model_name
        self.network_names = [f"{self.model_name}_{name}" for name in network_names]
        self.explainer =  Explainer(networks[0], ccnet_config, act_fn='tanh').to(device)
        self.reasoner =  Reasoner(networks[1], ccnet_config, act_fn=task_act_fn).to(device)
        self.producer =  Producer(networks[2], ccnet_config, act_fn="none").to(device)
        self.networks = [self.explainer, self.reasoner, self.producer]
        
        self.explain_size = ccnet_config.e_dim
        self.label_size = ccnet_config.y_dim
        self.device = device
        self.task_act_fn = task_act_fn
    
    def __set_train(self, train: bool):
        for network in self.networks:
            network.zero_grad()
            if train:
                network.train()
            else:
                network.eval()

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
            if self.use_seq_input:
                input_data = adjust_tensor_dim(input_data, target_dim=3)
            explanation = self._explain(input_data, padding_mask, batch_size = batch_size)
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
            if self.use_seq_input:
                original_dim = len(input_data.shape)
                input_data = adjust_tensor_dim(input_data, target_dim=3)
            explanation = self._explain(input_data, padding_mask, batch_size = batch_size)
            reasoned_output = self._reason(input_data, explanation, padding_mask, batch_size = batch_size)
            if self.use_seq_input:
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
            if self.use_seq_input:
                original_dim = len(input_data.shape)
                input_data = adjust_tensor_dim(input_data, target_dim=3)
            reasoned_output = self._reason(input_data, explanation, padding_mask, batch_size = batch_size)
            if self.use_seq_input:
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
            self.__set_train(True)

        return generated_output, condition_data

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
            if self.use_seq_input:
                original_dim = len(condition_data.shape)
                condition_data = adjust_tensor_dim(condition_data, target_dim=3)
                explanation = adjust_tensor_dim(explanation, target_dim=3)
            produced_output = self._produce(condition_data, explanation, padding_mask, batch_size = batch_size)
            if self.use_seq_input:
                produced_output = adjust_tensor_dim(produced_output, target_dim=original_dim)
            self.__set_train(True)
        return produced_output
    
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
            if self.use_seq_input:
                original_dim = len(input_data.shape)
                input_data = adjust_tensor_dim(input_data, target_dim=3)
            explanation = self._explain(input_data, padding_mask, batch_size = batch_size)
            inferred_output = self._reason(input_data, explanation, padding_mask, batch_size = batch_size)
            reconstructed_output = self._produce(inferred_output, explanation, padding_mask, batch_size = batch_size)
            if self.use_seq_input:
                reconstructed_output = adjust_tensor_dim(reconstructed_output, target_dim=original_dim)
            self.__set_train(True)
        return reconstructed_output
    
    def causal_generate(self, input_data, desired_target, padding_mask=None, batch_size=256):
        """
        Generates a new version of the input data by integrating counterfactual conditions. This function performs a
        three-step process: encoding the input, explaining to generate an explanation vector, and then producing
        an output conditioned on the counterfactual data. All steps are performed without updating the underlying model weights.

        Args:
            input_data (Tensor): The original input data tensor that you want to generate counterfactuals for.
            desired_target (Tensor): The counterfactual condition data tensor that specifies the desired state or conditions
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
            if self.use_seq_input:
                original_dim = len(input_data.shape)
                input_data = adjust_tensor_dim(input_data, desired_target_dim=3)
                desired_target = adjust_tensor_dim(desired_target, desired_target_dim=3)
            explanation = self._explain(input_data, padding_mask, batch_size=batch_size)
            causal_generated_data = self._produce(desired_target, explanation, padding_mask, batch_size=batch_size)
            if self.use_seq_input:
                causal_generated_data = adjust_tensor_dim(causal_generated_data, target_dim=original_dim)
            self.__set_train(True)
        return causal_generated_data

     