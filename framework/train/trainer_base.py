'''

Reference:


COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''

from nn.utils.init import set_random_seed
from framework.train.manager.optimization_manager import OptimizationManager
import torch

# Base class for trainers
class TrainerBase(OptimizationManager):
    def __init__(self, networks, training_params, optimization_params):
        self.train_iter = 0
        total_iterations = training_params.total_iters
        learning_params = [
            {'lr': optimization_params.learning_rate, 
             'decay_rate_100k': optimization_params.decay_rate_100k,
             'scheduler_type': optimization_params.scheduler_type, 
             'clip_grad_range': optimization_params.clip_grad_range, 
             'max_grad_norm': optimization_params.max_grad_norm}
            for _ in networks
        ]
        OptimizationManager.__init__(self, networks, learning_params, total_iterations)
        self.networks = networks

    def set_train(self, train: bool):
        for network in self.networks:
            network.zero_grad()
            if train:
                network.train()
            else:
                network.eval()                

    def update_step(self):
        self.clip_gradients()    
        self.update_optimizers()    
        self.update_schedulers()
        self.update_seed()

    def reset_seed(self):
        set_random_seed(self.train_iter)
        
    def update_seed(self):
        self.train_iter += 1
        set_random_seed(self.train_iter)

    def backwards(self, networks, network_errors):
        """
        This function orchestrates the backward pass for multiple models, enabling targeted backpropagation 
        based on specific errors. Designed for a broad array of machine learning applications, it allows for 
        precise error correction and model refinement. By processing errors independently for each model, the 
        framework ensures accurate updates without cross-interference, crucial for complex tasks involving 
        multiple interrelated models.

        Step-by-step Explanation:
        1. Initially, disable gradient computations for all models to avoid unintended updates.
        2. Iteratively for each model:
        - Activate gradient calculations to enable targeted backpropagation.
        - Clear existing gradients to prevent accumulation from prior updates.
        - Backpropagate the designated errors, selectively retaining the computation graph as needed.
        - Deactivate gradient updates post-error processing to maintain isolation.
        3. Reactivate gradients for all models, preparing them for future learning iterations.

        This meticulous approach enhances model specialization and the efficacy of the learning process by 
        ensuring that updates are directly aligned with specific error signals.

        :param models: List of models for backpropagation.
        :param model_errors: Corresponding errors for each model, dictating the backpropagation process.
        """
        # Implementation details follow
        num_network = len(networks)
        # Temporarily disable gradient computation for all networks
        for network in networks:
            network.requires_grad_(False)        
            
        for net_idx, (network, error) in enumerate(zip(networks, network_errors)):
            # Enable gradient computation for the current network
            network.requires_grad_(True)
            # Zero out the gradient for all networks starting from the current network
            for n in networks[net_idx:]:
                n.zero_grad()
            # Decide whether to retain the computation graph based on the network and error index
            retain_graph = (net_idx < num_network - 1)
            # Apply the discounted gradients in the backward pass
            error.backward(torch.ones_like(error), retain_graph=retain_graph)
            # Prevent gradient updates for the current network after its errors have been processed
            network.requires_grad_(False)
            
        # Restore gradient computation capability for all networks for potential future forward passes
        for network in networks:
            network.requires_grad_(True)
        return