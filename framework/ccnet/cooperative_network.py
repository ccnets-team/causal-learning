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
    def __init__(self, model_params, state_size, label_size, explain_size, device, 
                 encoder = None):
        # Initialize model names and configurations.
        explainer_params = model_params.explainer_params
        reasoner_params = model_params.reasoner_params
        producer_params = model_params.producer_params
        
        explainer_network = model_params.explainer_network
        reasoner_network = model_params.reasoner_network
        producer_network = model_params.producer_network

        self.encoder = encoder

        # Create and move each component model to the specified device.
        self.network_names = ["explainer", "reasoner", "producer"]
        self.explainer =  Explainer(explainer_network, explainer_params, [state_size], explain_size, act_fn="layer_norm").to(device)
        self.reasoner =  Reasoner(reasoner_network, reasoner_params, [state_size], explain_size, label_size, act_fn="sigmoid").to(device)
        self.producer =  Producer(producer_network, producer_params, label_size, explain_size, [state_size], act_fn="none").to(device)
        self.networks = [self.explainer, self.reasoner, self.producer]
        
        self.explain_size = explain_size
        self.state_size = state_size
        self.label_size = label_size
        self.__device = device

    def __encode(self, x):
        if self.encoder is not None:
            x = self.encoder.encode(x)
        return x

    def __decode(self, x):
        if self.encoder is not None:
            x = self.encoder.decode(x)
        return x
            
    # Generate explanations for the input x without updating the model.
    def explain(self, x):
        with torch.no_grad():
            code = self.__encode(x)
            code = adjust_tensor_dim(code, target_dim=3)
            e = self.explainer(code)
        return e

    # A streamlined process for inferring y from x using both the explainer and reasoner models without updating them.
    def infer(self, x):
        with torch.no_grad():
            x_code = self.__encode(x)
            original_dim_len = len(x_code.shape)
            x_code = adjust_tensor_dim(x_code, target_dim=3)
            e = self.explainer(x_code)
            y_code = self.reasoner(x_code, e)
            y_code = adjust_tensor_dim(y_code, target_dim=original_dim_len)
        return y_code
    
    # Use the explanations and the input to reason about the output y without updating the model.
    def reason(self, x, e):
        with torch.no_grad():
            x = self.__encode(x)
            original_dim_len = len(x.shape)
            x = adjust_tensor_dim(x, target_dim=3)
            y_code = self.reasoner(x, e)
            y_code = adjust_tensor_dim(y_code, target_dim=original_dim_len)
        return y_code

    # Generate new data based on y without updating the producer model.
    def generate(self, y):
        with torch.no_grad():
            original_dim_len = len(y.shape)
            y = adjust_tensor_dim(y, target_dim=3)
            random_explain = torch.randn(y.size(0), self.explain_size).to(self.__device)   
            code_generated = self.producer(y, random_explain)
            code_generated = adjust_tensor_dim(code_generated, target_dim=original_dim_len)
            x_generated = self.__decode(code_generated)
        return x_generated

    # Generate new data based on y without updating the producer model.
    def produce(self, y, e):
        with torch.no_grad():
            original_dim_len = len(y.shape)
            y = adjust_tensor_dim(y, target_dim=3)
            e = adjust_tensor_dim(e, target_dim=3)
            code_produced = self.producer(y, e)
            code_produced = adjust_tensor_dim(code_produced, target_dim=original_dim_len)
            x_produced = self.__decode(code_produced)
        return x_produced
    
    # Reconstruct the input x by first explaining, then reasoning, 
    # and finally producing the output, all without updating the models.
    def reconstruct(self, x):
        with torch.no_grad():
            x_code = self.__encode(x)
            original_dim_len = len(x_code.shape)
            x_code = adjust_tensor_dim(x_code, target_dim=3)
            e = self.explainer(x_code)
            y_inferred = self.reasoner(x_code, e)
            code_reconstructed = self.producer(y_inferred, e)
            code_reconstructed = adjust_tensor_dim(code_reconstructed, target_dim=original_dim_len)
            x_reconstructed = self.__decode(code_reconstructed)
        return x_reconstructed