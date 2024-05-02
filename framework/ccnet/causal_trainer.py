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
from framework.train.utils.metrics_tracker import create_causal_training_metrics
from framework.ccnet.cooperative_network import CooperativeNetwork
from framework.train.trainer_base import TrainerBase

class CausalTrainer(TrainerBase):
    def __init__(self, ccnet: CooperativeNetwork, training_params, optimization_params):
        TrainerBase.__init__(self, ccnet.networks, training_params, optimization_params)
        self.explainer, self.reasoner, self.producer = self.networks        
    
    def train_models(self, state, label, padding_mask=None):
        # Set the models to training mode and perform the forward pass.
        self.set_train(train=True)
        ################################  Forward Pass  ################################################
        explain = self.explainer(state, padding_mask)
        inferred_label = self.reasoner(state, explain, padding_mask)
        
        # reset random seed for internal noise factor
        self.reset_seed()
        generated_state = self.producer(label, explain, padding_mask)
        
        # reset random seed for internal noise factor
        self.reset_seed()
        reconstructed_state = self.producer(inferred_label, explain.detach(), padding_mask)

        ################################  Prediction Losses  ###########################################
        # Calculate prediction losses for inference, generation, and reconstruction.
        inference_loss = self.loss_fn(reconstructed_state, generated_state)
        generation_loss = self.loss_fn(generated_state , state)
        reconstruction_loss = self.loss_fn(reconstructed_state, state)

        ################################  Model Losses  ################################################
        # Calculate model errors based on the losses.
        explainer_error = self.error_fn(inference_loss + generation_loss, reconstruction_loss)
        reasoner_error = self.error_fn(reconstruction_loss + inference_loss, generation_loss)
        producer_error = self.error_fn(generation_loss + reconstruction_loss, inference_loss)

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

    def loss_fn(self, predict, target):
        discrepancy = (predict - target.detach()).abs()
        prediction_loss = discrepancy.mean(dim=-1, keepdim = True)
        return prediction_loss
    
    def error_fn(self, predict, target):
        discrepancy = (predict - target.detach()).abs()
        cooperative_error = discrepancy.mean(dim = 0) 
        return cooperative_error