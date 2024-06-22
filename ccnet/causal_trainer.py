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
from ccnet.causal_cooperative_net import CausalCooperativeNet as CCNet
from ccnet.trainer_base import TrainerBase

class CausalTrainer(TrainerBase):
    def __init__(self, ccnet: CCNet, ccnet_config, train_config, opt_config):
        TrainerBase.__init__(self, ccnet.networks, ccnet_config, train_config, opt_config, ccnet.device)
        self.explainer, self.reasoner, self.producer = self.networks  
        self.network_names = ccnet.network_names
    
    def train_models(self, observation, label, padding_mask=None):
        # Set the models to training mode and perform the forward pass.
        self.set_train(train=True)
        
        ################################  Forward Pass  ################################################
        explain = self.explainer(observation, padding_mask)
        inferred_label = self.reasoner(observation, explain, padding_mask)
        
        # reset random seed for internal noise factor
        self.reset_seed()
        generated_observation = self.producer(label, explain, padding_mask)
        
        # reset random seed for internal noise factor
        self.reset_seed()
        reconstructed_observation = self.producer(inferred_label, explain.detach(), padding_mask)

        ################################  Prediction Losses  ###########################################
        # Calculate prediction losses for inference, generation, and reconstruction.
        inference_loss = self.loss_fn(reconstructed_observation, generated_observation, padding_mask)
        generation_loss = self.loss_fn(generated_observation, observation, padding_mask)
        reconstruction_loss = self.loss_fn(reconstructed_observation, observation, padding_mask)

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