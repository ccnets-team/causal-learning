'''
    Causal learning implementation in PyTorch.
    Author:
        PARK, JunHo, junho@ccnets.org
    Reference:
        https://www.linkedin.com/feed/update/urn:li:activity:7127983766643347456
        https://patents.google.com/patent/US20230359867A1/
        https://patents.google.com/patent/KR102656365B1/

    COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''

from tools.metric_tracker import create_causal_training_metrics
from framework.ccnet.cooperative_network import CooperativeNetwork
from framework.train.trainer_base import TrainerBase

class CausalTrainer(TrainerBase):
    def __init__(self, ccnet: CooperativeNetwork, algorithm_params, optimization_params, task_type):
        TrainerBase.__init__(self, ccnet.networks, algorithm_params, optimization_params, task_type, ccnet.device)
        self.explainer, self.reasoner, self.producer = self.networks  
        self.network_names = ccnet.network_names
        self.use_gpt = ccnet.use_gpt
    
    def train_models(self, state, label, padding_mask=None):
        # Set the models to training mode and perform the forward pass.
        self.set_train(train=True)
        
        input_state, target_state, label = self.prepare_data(state, label)
        
        ################################  Forward Pass  ################################################
        explain = self.explainer(input_state, padding_mask)
        inferred_label = self.reasoner(input_state, explain, padding_mask)
        
        # reset random seed for internal noise factor
        self.reset_seed()
        generated_state = self.producer(label, explain, padding_mask)
        
        # reset random seed for internal noise factor
        self.reset_seed()
        reconstructed_state = self.producer(inferred_label, explain.detach(), padding_mask)

        ################################  Prediction Losses  ###########################################
        # Calculate prediction losses for inference, generation, and reconstruction.
        inference_loss = self.loss_fn(reconstructed_state, generated_state, padding_mask)
        generation_loss = self.loss_fn(generated_state, target_state, padding_mask)
        reconstruction_loss = self.loss_fn(reconstructed_state, target_state, padding_mask)

        ################################  Model Losses  ################################################
        # Calculate model errors based on the losses.
        explainer_error = self.error_fn(inference_loss + generation_loss, reconstruction_loss, padding_mask)
        reasoner_error = self.error_fn(reconstruction_loss + inference_loss, generation_loss, padding_mask)
        producer_error = self.error_fn(generation_loss + reconstruction_loss, inference_loss, padding_mask)

        ################################  Backward Pass  ###############################################
        # Perform the backward pass and update the models.
        self.backwards(
            [self.explainer, self.reasoner, self.producer], 
            [explainer_error, reasoner_error, producer_error]
            )
        ################################  Model Update  ###############################################
        # Update optimizers and schedulers, and reset the random seed.
        self.update_step()

        # Calculate and return the mean losses and errors.
        metrics = create_causal_training_metrics(
            inference_loss = inference_loss,
            generation_loss = generation_loss,
            reconstruction_loss = reconstruction_loss,
            explainer_error = explainer_error, 
            reasoner_error = reasoner_error, 
            producer_error = producer_error,
            padding_mask = padding_mask
        )
        return metrics

    def loss_fn(self, predict, target, padding_mask=None):
        """Calculate the prediction loss, optionally excluding padded data."""
        discrepancy = (predict - target.detach()).abs()
        if padding_mask is not None:
            discrepancy *= padding_mask
            
        """Calculate mean of data, use mask if provided to exclude padded data."""
        if padding_mask is not None:
            expanded_mask = padding_mask.expand_as(discrepancy)
            if self.use_gpt:
                return discrepancy.sum(dim=-1, keepdim=True) / expanded_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            else:
                return discrepancy.view(discrepancy.size(0), -1).sum(dim=1, keepdim=True) / expanded_mask.view(expanded_mask.size(0), -1).sum(dim=1, keepdim=True).clamp_min(1)
        else:
            if self.use_gpt:
                return discrepancy.mean(dim=-1, keepdim=True)
            else:
                return discrepancy.view(discrepancy.size(0), -1).mean(dim=1, keepdim=True)
        
    def error_fn(self, predict, target, padding_mask=None):

        if self.error_function == 'mse':
            discrepancy = (predict - target.detach()).square()
        else:
            discrepancy = (predict - target.detach()).abs()
        
        # Compute the mean error, considering only the non-padded data
        if padding_mask is not None:
            discrepancy *= padding_mask
            expanded_mask = padding_mask.expand_as(discrepancy)
            cooperative_error = discrepancy.sum(dim = 0, keepdim = True) / expanded_mask.sum(dim = 0, keepdim = True).clamp_min(1)
        else:
            cooperative_error = discrepancy.mean(dim = 0, keepdim = True)
        
        return cooperative_error
    
    def update_step(self):
        self.clip_gradients()    
        self.update_optimizers()    
        self.update_schedulers()
        self.update_seed()
