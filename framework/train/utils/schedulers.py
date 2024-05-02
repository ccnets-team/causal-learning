
from torch.optim.lr_scheduler import _LRScheduler, CyclicLR

STEPS_100K = 100000  # Represents the number of steps over which decay is applied
LR_CYCLE_SIZE = 20000
DECAY_RATE_100K = 0.1  # Represents the decay rate over 100k steps

class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, decay_rate_100k, last_epoch=-1):
        self.total_steps = total_steps
        # Calculate the final learning rate multiplier considering the total steps
        self.final_lr_multiplier = pow(decay_rate_100k, total_steps / STEPS_100K)
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Calculate the proportional step for the current epoch relative to total_steps
        proportional_step = min(self.last_epoch / self.total_steps, 1)
        # Linearly interpolate the learning rate towards the final_lr_multiplier
        lr_decay = 1 - proportional_step + (self.final_lr_multiplier * proportional_step)
        return [base_lr * lr_decay for base_lr in self.base_lrs]