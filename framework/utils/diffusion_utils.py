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
    def __init__(self, T=1000, beta_start=0.0001, beta_end=0.02):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = linear_beta_schedule(self.T, self.beta_start, self.beta_end)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)

    def diffuse(self, state, t):
        """
        Diffuse the state to a noisy version at timestep t and t+1.

        Args:
            state (torch.Tensor): The original state.
            t (int): The current timestep.

        Returns:
            tuple: Noisy input state and target state.
        """
        alpha_t = self.alpha_cumprod[t]
        beta_t = self.betas[t]

        noise_t = torch.normal(0, 1, state.shape)
        input_state = torch.sqrt(alpha_t) * state + torch.sqrt(beta_t) * noise_t
        
        if t + 1 < self.T:
            alpha_t_plus_1 = self.alpha_cumprod[t + 1]
            beta_t_plus_1 = self.betas[t + 1]
            noise_t_plus_1 = torch.normal(0, 1, state.shape)
            target_state = torch.sqrt(alpha_t_plus_1) * state + torch.sqrt(beta_t_plus_1) * noise_t_plus_1
        else:
            target_state = input_state  # Edge case: if t is the last timestep

        return input_state, target_state
