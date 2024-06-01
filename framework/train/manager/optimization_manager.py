import torch
import torch.optim as optim

LR_CYCLE_SIZE = 20000
STEPS_100K = 100000  # Represents the number of steps over which decay is applied

class OptimizationManager:
    def __init__(self, networks, learning_params):

        self.optimizers = []
        self.schedulers = []
        self.clip_grad_ranges = []
        self.max_grad_norms = []
        self.current_lrs = []
        self.initial_lrs = []

        self.setup_optimization(networks, learning_params)
        self.__networks = networks

    def setup_optimization(self, networks, learning_params):
        for network, params in zip(networks, learning_params):
            optimizer = self.create_optimizer(network, params['lr'])
            scheduler = self.create_scheduler(optimizer, params)
            
            self.current_lrs.append(params['lr'])
            self.initial_lrs.append(params['lr'])
            self.clip_grad_ranges.append(params['clip_grad_range'])
            self.max_grad_norms.append(params['max_grad_norm'])
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)

    def create_optimizer(self, network, lr):
        return optim.Adam(network.parameters(), lr=lr, betas=(0.9, 0.999))

    def create_scheduler(self, optimizer, params):
        scheduler_type = params['scheduler_type']
        decay_rate = params['decay_rate_100k']
        if scheduler_type == 'exponential':
            gamma = pow(decay_rate, 1/STEPS_100K)
            return optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def get_lr(self):
        # Get the first learning rate from each parameter group in each optimizer
        current_lrs = [optimizer.param_groups[0]['lr'] for optimizer in self.optimizers]
        # Take only the first learning rate from the list of optimizers
        first_lr = current_lrs[0] if current_lrs else 0.0
        return first_lr
    
    def clip_gradients(self):
        for idx, net in enumerate(self.__networks):
            # Check if net is an instance of LearnableTD
            clip_grad_range = self.clip_grad_ranges[idx]
            max_grad_norm = self.max_grad_norms[idx]
            # Handling for other network types
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            if clip_grad_range is not None:
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_grad_range)

    def update_optimizers(self):
        for opt in self.optimizers:
            opt.step()
            opt.zero_grad()
                
    def update_schedulers(self):
        for sc in self.schedulers:
            sc.step()
            