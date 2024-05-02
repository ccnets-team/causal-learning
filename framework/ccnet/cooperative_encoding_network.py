'''
    Author:
        PARK, JunHo, junho@ccnets.org
'''
import torch
from .roles.explainer import Explainer
from .roles.reasoner import Reasoner
from .roles.producer import Producer

class CooperativeEncodingNetwork:
    def __init__(self, model_params, obs_shape, stoch_size, det_size, device):
        # Initialize model names and configurations.
        encoding_networks = model_params.encoding_networks
        encoding_params = model_params.encoding_params
        encoding_params.obs_shape = obs_shape
        encoding_params.z_dim = det_size
        encoding_params.condition_dim = stoch_size
            
        self.explainer = Explainer(encoding_networks[0], encoding_params, obs_shape, det_size, act_fn="layer_norm").to(device)
        self.reasoner = Reasoner(encoding_networks[1], encoding_params, obs_shape, det_size, stoch_size, act_fn="layer_norm").to(device)
        self.producer = Producer(encoding_networks[2], encoding_params, stoch_size, det_size, obs_shape, act_fn="none").to(device)

        self.network_names = ["explainer", "reasoner", "producer"]
        self.networks = [self.explainer, self.reasoner, self.producer]
        
        self.obs_shape = obs_shape
        self.det_size = det_size
        self.stoch_size = stoch_size

    def encode(self, x):
        with torch.no_grad():
            e = self.explainer(x)
            f = self.reasoner(x, e)
            z = torch.cat([f, e], dim = -1)
        return z

    def decode(self, z):
        with torch.no_grad():
            f = z[..., :self.stoch_size]
            e = z[..., self.stoch_size:]
            x = self.producer(f, e)
        return x