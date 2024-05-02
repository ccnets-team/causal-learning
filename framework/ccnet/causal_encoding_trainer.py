'''
    Unsupervised Causal learning implementation in PyTorch.
    
    Reference:
        "Cooperative architecture for unsupervised learning of causal relationships in data generation"
        https://patents.google.com/patent/WO2023224428A1/
        
    Author:
        PARK, JunHo, junho@ccnets.org
'''
import torch
from framework.ccnet.cooperative_encoding_network import CooperativeEncodingNetwork
from framework.train.trainer_base import TrainerBase
from framework.train.utils.metrics_tracker import create_causal_training_metrics

class CausalEncodingTrainer(TrainerBase):
    def __init__(self, encoder: CooperativeEncodingNetwork, training_params, optimization_params):
        TrainerBase.__init__(self, encoder.networks, training_params, optimization_params)
        self.explainer, self.reasoner, self.producer = self.networks        

    def train_models(self, input_observation, **kwargs):
        self.set_train(train=True)

        causal_explain = self.explainer(input_observation)
        random_explain1 = torch.randn_like(causal_explain)
        random_explain2 = torch.randn_like(causal_explain)
        
        causal_feature = self.reasoner(input_observation, causal_explain)
        stochastic_feature = self.reasoner(input_observation, random_explain1)
        ###################################################
        self.reset_seed()
        cc_generated_observation = self.producer(causal_feature, causal_explain.detach())
        self.reset_seed()
        cs_generated_observation = self.producer(causal_feature, random_explain2)
        self.reset_seed()
        sc_generated_observation = self.producer(stochastic_feature.detach(), causal_explain)
        self.reset_seed()
        ss_generated_observation = self.producer(stochastic_feature, random_explain2).detach()
        ###################################################
        reconstruction_loss = self.loss_fn(cc_generated_observation, input_observation)

        cost_ds = self.cost_fn(cs_generated_observation, ss_generated_observation)
        cost_sc = self.cost_fn(sc_generated_observation, input_observation)
        recognition_loss = self.loss_fn(cost_ds, cost_sc)
        ###################################################
        
        cost_sd = self.cost_fn(sc_generated_observation, ss_generated_observation)
        cost_cs = self.cost_fn(cs_generated_observation, input_observation)
        generation_loss = self.loss_fn(cost_sd, cost_cs)
        ###################################################

        explainer_error = self.error_fn(recognition_loss + generation_loss, reconstruction_loss)
        reasoner_error = self.error_fn(reconstruction_loss, generation_loss - recognition_loss)
        producer_error = self.error_fn(reconstruction_loss, recognition_loss - generation_loss)
        ###################################################

        self.backwards(
            [self.explainer, self.reasoner, self.producer], 
            [explainer_error, reasoner_error, producer_error]
            )
        
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
        cost = (predict - target.detach()).abs()
        return cost

    def loss_fn(self, predict, target):
        prediction_loss = (predict - target.detach()).abs().mean()
        return prediction_loss
    
    def error_fn(self, predict, target):
        cooperative_error = (predict - target.detach()).abs()
        return cooperative_error
