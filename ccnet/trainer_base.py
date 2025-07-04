'''

Reference:


COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''

from nn.utils.init_layer import set_random_seed
from ccnet.manager.optimization_manager import OptimizationManager
import torch

# Base class for trainers
class TrainerBase(OptimizationManager):
    def __init__(self, networks, ccnet_config, train_config, opt_config, device):
        self.train_iter = 0
        learning_params = [
            {'lr': opt_config.learning_rate, 
             'decay_rate_100k': opt_config.decay_rate_100k,
             'scheduler_type': opt_config.scheduler_type, 
             'clip_grad_range': opt_config.clip_grad_range, 
             'max_grad_norm': opt_config.max_grad_norm}
            for _ in networks
        ]
        OptimizationManager.__init__(self, networks, learning_params)
        self.networks = networks
        self.initial_lr = opt_config.learning_rate
        self.error_function = train_config.error_function
        
        self.device = device
        self.task_type = ccnet_config.task_type
        self.obs_shape = ccnet_config.obs_shape

    def set_train(self, train: bool):
        for network in self.networks:
            network.zero_grad()
            if train:
                network.train()
            else:
                network.eval()                

    def reset_seed(self):
        set_random_seed(self.train_iter)
        
    def update_seed(self):
        self.train_iter += 1
        set_random_seed(self.train_iter)

    def update_step(self):
        self.clip_gradients()    
        self.update_optimizers()    
        self.update_schedulers()
        self.update_seed()

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

    def cost_fn(self, predict, target):
        # Calculate path costs derived from the absolute difference between predictions and targets
        path_cost = (predict - target.detach()).abs()
        return path_cost
    
    def loss_fn(self, predict, target, padding_mask = None):
        # Compute the absolute discrepancy between predictions and actual values
        absolute_discrepancy = (predict - target.detach()).abs()
        
        # Flatten the observation shape for feature reduction while keeping batch or sequence dimensions intact
        preserved_shape = absolute_discrepancy.shape[:len(absolute_discrepancy.shape) - len(self.obs_shape)]
        flattened_discrepancy = absolute_discrepancy.reshape(*preserved_shape, -1)
        
        if padding_mask is not None:
            # Apply the padding mask to the flattened discrepancy tensor
            flattened_discrepancy *= padding_mask
            
        reduced_tensor = flattened_discrepancy.mean(dim=-1, keepdim = True)
        return reduced_tensor
        
    def error_fn(self, predict, target, padding_mask=None):
        # Compute the discrepancy based on the specified error function
        if self.error_function == 'mse':
            discrepancy_tensor = (predict - target.detach()).square()
            
        elif self.error_function == 'rmsle':
            # predict = torch.clamp(predict, min=0)
            # target = torch.clamp(target, min=0)

            # Compute logs safely
            log_pred = torch.log1p(predict)
            log_true = torch.log1p(target)

            # Calculate the squared differences of the logarithmic values
            squared_log_error = torch.square(log_pred - log_true)

            # Mean squared log error
            mean_squared_log_error = torch.mean(squared_log_error)

            # Root of the mean squared log error to get RMSLE
            discrepancy_tensor = torch.sqrt(mean_squared_log_error)
            
        else:
            discrepancy_tensor = (predict - target.detach()).abs()
        
        if padding_mask is not None:
            # Compute the sum of the input tensor, considering only the non-padded data
            reduced_tensor = discrepancy_tensor.sum(dim=0, keepdim=True) / padding_mask.sum(dim=0, keepdim=True).clamp_min(1)
        else:
            # Compute the mean of the input tensor
            reduced_tensor = discrepancy_tensor.mean(dim=0, keepdim=True)
        
        return reduced_tensor