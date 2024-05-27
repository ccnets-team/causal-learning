'''
    Unsupervised Causal learning implementation in PyTorch.
    
    Reference:
        "Cooperative architecture for unsupervised learning of causal relationships in data generation"
        https://patents.google.com/patent/WO2023224428A1/
        
    Author:
        PARK, JunHo, junho@ccnets.org
'''
import torch
from framework.ccnet.cooperative_encoding_network import CooperativeEncodingNetwork as Encoder
from framework.train.trainer_base import TrainerBase
from tools.metric_tracker import create_causal_training_metrics

class CausalEncodingTrainer(TrainerBase):
    """
    Trainer for the Causal Encoding Network, handling the training process across
    the network's components: Explainer, Reasoner, and Producer.
    """
    def __init__(self, encoder: Encoder, training_params, algorithm_params, optimization_params):
        """
        Initialize the trainer with an encoder network, training parameters, and optimization settings.

        Parameters:
        - encoder: Instance of CooperativeEncodingNetwork used to encode and decode data.
        - training_params: Dictionary containing parameters specific to training (e.g., batch size, epochs).
        - optimization_params: Dictionary containing parameters specific to optimization (e.g., learning rate).
        """        
        TrainerBase.__init__(self, encoder.networks, training_params, algorithm_params, optimization_params, encoder.device)
        self.explainer, self.reasoner, self.producer = self.networks        
        self.network_names = encoder.network_names
        self.layer_norm = torch.nn.LayerNorm(encoder.stoch_size, elementwise_affine=False).to(encoder.device)

    def train_models(self, observation, padding_mask=None):
        """
        Train the models using input observations. Generates multiple interpretations and compares them.

        Parameters:
        - input_observation: Input data for training.
        
        Returns:
        - metrics: Dictionary containing tracked metrics (losses and errors).
        """
        self.set_train(train=True)
        
        input_observation, target_observation = self.prepare_data(observation)

        ################################  Forward Pass  ########################################
        # Generate explanations and features.
        causal_explain = self.explainer(input_observation, padding_mask)
        random_explain = torch.randn_like(causal_explain)
        
        causal_feature = self.reasoner(input_observation, causal_explain, padding_mask)
        stochastic_feature = self.reasoner(input_observation, random_explain, padding_mask)
        
        # Reset random seed for reproducibility before each generation.        
        self.reset_seed()
        cc_generated_observation = self.producer(causal_feature, causal_explain.detach(), padding_mask)
        self.reset_seed()
        cs_generated_observation = self.producer(causal_feature, random_explain, padding_mask)
        self.reset_seed()
        sc_generated_observation = self.producer(stochastic_feature.detach(), causal_explain, padding_mask)
        self.reset_seed()
        ss_generated_observation = self.producer(stochastic_feature, random_explain, padding_mask).detach()

        ################################  Path Costs  ###########################################
        cost_ds = self.cost_fn(cs_generated_observation, ss_generated_observation)
        cost_sc = self.cost_fn(sc_generated_observation, observation)
        cost_sd = self.cost_fn(sc_generated_observation, ss_generated_observation)
        cost_cs = self.cost_fn(cs_generated_observation, observation)

        ################################  Causal Losses  ########################################
        recognition_loss = self.loss_fn(cost_ds, cost_sc, padding_mask)
        generation_loss = self.loss_fn(cost_sd, cost_cs, padding_mask)
        reconstruction_loss = self.loss_fn(cc_generated_observation, observation, padding_mask)

        ################################  Model Errors  #########################################
        explainer_error = self.error_fn(recognition_loss + generation_loss, reconstruction_loss, padding_mask)
        reasoner_error = self.error_fn(reconstruction_loss, generation_loss - recognition_loss, padding_mask)
        producer_error = self.error_fn(reconstruction_loss, recognition_loss - generation_loss, padding_mask)

        ################################  Backward Pass  ########################################
        self.backwards(
            [self.explainer, self.reasoner, self.producer], 
            [explainer_error, reasoner_error, producer_error]
            )
        
        ################################  Model Update  #########################################
        self.update_step()

        # Calculate and return the mean losses and errors.
        metrics = create_causal_training_metrics(
            inference_loss = recognition_loss,
            generation_loss = generation_loss,
            reconstruction_loss = reconstruction_loss,
            explainer_error = explainer_error, 
            reasoner_error = reasoner_error, 
            producer_error = producer_error
        )
        return metrics
    
    def cost_fn(self, predict, target):
        """
        Calculate absolute difference between predictions and targets.

        Parameters:
        - predict: Predicted values.
        - target: Target values.
        
        Returns:
        - cost: Absolute difference.
        """        
        path_cost = (predict - target.detach()).abs()
        return path_cost

    def loss_fn(self, predict, target, padding_mask = None):
        """
        Compute mean absolute loss between predictions and targets.

        Parameters:
        - predict: Predicted values.
        - target: Target values.
        
        Returns:
        - causal_loss: Mean absolute difference.
        """        
        causal_loss = (predict - target.detach()).abs().mean()
        return causal_loss
    
    def error_fn(self, predict, target, padding_mask = None):
        """
        Compute absolute error between the predicted and target values.

        Parameters:
        - predict: The calculated prediction or computed value.
        - target: The target or expected value.

        Returns:
        - cooperative_error: Absolute error calculated as the absolute difference between prediction and target.
        """        
        cooperative_error = (predict - target.detach()).abs()
        return cooperative_error

    def update_step(self):
        self.update_optimizers()    
        self.update_schedulers()
        self.update_seed()
