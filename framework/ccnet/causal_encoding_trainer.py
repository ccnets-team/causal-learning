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
from tools.metrics_tracker import create_causal_training_metrics

class CausalEncodingTrainer(TrainerBase):
    """
    Trainer for the Causal Encoding Network, handling the training process across
    the network's components: Explainer, Reasoner, and Producer.
    """
    def __init__(self, encoder: Encoder, training_params, optimization_params):
        """
        Initialize the trainer with an encoder network, training parameters, and optimization settings.

        Parameters:
        - encoder: Instance of CooperativeEncodingNetwork used to encode and decode data.
        - training_params: Dictionary containing parameters specific to training (e.g., batch size, epochs).
        - optimization_params: Dictionary containing parameters specific to optimization (e.g., learning rate).
        """        
        TrainerBase.__init__(self, encoder.networks, training_params, optimization_params)
        self.explainer, self.reasoner, self.producer = self.networks        

    def train_models(self, input_observation, **kwargs):
        """
        Train the models using input observations. Generates multiple interpretations and compares them.

        Parameters:
        - input_observation: Input data for training.
        
        Returns:
        - metrics: Dictionary containing tracked metrics (losses and errors).
        """
        self.set_train(train=True)

        ################################  Forward Pass  ########################################
        # Generate explanations and features.
        causal_explain = self.explainer(input_observation)
        random_explain1 = torch.randn_like(causal_explain)
        random_explain2 = torch.randn_like(causal_explain)
        
        causal_feature = self.reasoner(input_observation, causal_explain)
        stochastic_feature = self.reasoner(input_observation, random_explain1)
        
        # Reset random seed for reproducibility before each generation.        
        self.reset_seed()
        cc_generated_observation = self.producer(causal_feature, causal_explain.detach())
        self.reset_seed()
        cs_generated_observation = self.producer(causal_feature, random_explain2)
        self.reset_seed()
        sc_generated_observation = self.producer(stochastic_feature.detach(), causal_explain)
        self.reset_seed()
        ss_generated_observation = self.producer(stochastic_feature, random_explain2).detach()

        ################################  Path Costs  ###########################################
        cost_ds = self.cost_fn(cs_generated_observation, ss_generated_observation)
        cost_sc = self.cost_fn(sc_generated_observation, input_observation)
        cost_sd = self.cost_fn(sc_generated_observation, ss_generated_observation)
        cost_cs = self.cost_fn(cs_generated_observation, input_observation)

        ################################  Causal Losses  ########################################
        recognition_loss = self.loss_fn(cost_ds, cost_sc)
        generation_loss = self.loss_fn(cost_sd, cost_cs)
        reconstruction_loss = self.loss_fn(cc_generated_observation, input_observation)

        ################################  Model Errors  #########################################
        explainer_error = self.error_fn(recognition_loss + generation_loss, reconstruction_loss)
        reasoner_error = self.error_fn(reconstruction_loss, generation_loss - recognition_loss)
        producer_error = self.error_fn(reconstruction_loss, recognition_loss - generation_loss)

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
        cost = (predict - target.detach()).abs()
        return cost

    def loss_fn(self, predict, target):
        """
        Compute mean absolute loss between predictions and targets.

        Parameters:
        - predict: Predicted values.
        - target: Target values.
        
        Returns:
        - prediction_loss: Mean absolute difference.
        """        
        prediction_loss = (predict - target.detach()).abs().mean()
        return prediction_loss
    
    def error_fn(self, predict, target):
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
