import torch

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Create a linear schedule for beta values from beta_start to beta_end.
    
    Args:
        timesteps (int): Number of timesteps T.
        beta_start (float): The starting value of beta.
        beta_end (float): The ending value of beta.
    
    Returns:
        torch.Tensor: Array of beta values for each timestep.
    """
    return torch.linspace(beta_start, beta_end, timesteps)

class NoiseDiffuser:
    def __init__(self, T=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = linear_beta_schedule(self.T, self.beta_start, self.beta_end).to(device)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0).to(device)

    def diffuse(self, state, t):
        """
        Diffuse the state to a noisy version at timestep t and t+1.

        Args:
            state (torch.Tensor): The original state.
            t (torch.Tensor): The current timesteps for each sample in the batch.

        Returns:
            tuple: Noisy input state and target state.
        """
        device = state.device
        batch_size = state.shape[0]
        num_dims = len(state.shape[1:])

        # Get alpha_t and beta_t for each sample in the batch
        alpha_t = self.alpha_cumprod[t].view(batch_size, *([1] * num_dims)).to(device)
        beta_t = self.betas[t].view(batch_size, *([1] * num_dims)).to(device)

        noise_t = torch.normal(0, 1, state.shape, device=device)
        target_state = torch.sqrt(alpha_t) * state + torch.sqrt(beta_t) * noise_t
        
        # Prepare the next timestep
        next_t = torch.clamp(t + 1, max=self.T - 1)
        alpha_t_plus_1 = self.alpha_cumprod[next_t].view(batch_size, *([1] * num_dims)).to(device)
        beta_t_plus_1 = self.betas[next_t].view(batch_size, *([1] * num_dims)).to(device)
        noise_t_plus_1 = torch.normal(0, 1, state.shape, device=device)
        input_state = torch.sqrt(alpha_t_plus_1) * state + torch.sqrt(beta_t_plus_1) * noise_t_plus_1

        return input_state, target_state