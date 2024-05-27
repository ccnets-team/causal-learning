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

def get_alpha_beta_t(t, alpha_cumprod, betas, device, num_dims):
    alpha_t = alpha_cumprod[t].view(t.shape[0], *([1] * num_dims)).to(device)
    beta_t = betas[t].view(t.shape[0], *([1] * num_dims)).to(device)
    return alpha_t, beta_t

def add_noise(x, alpha_t, beta_t, device):
    noise_t = torch.normal(0, 1, x.shape, device=device)
    return torch.sqrt(alpha_t) * x + torch.sqrt(beta_t) * noise_t

def generate_target_state(state, t, alpha_cumprod, betas, device):
    """
    Generate the target state by adding noise at timestep t.

    Args:
        state (torch.Tensor): The original state.
        t (torch.Tensor): Timesteps tensor.
        alpha_cumprod (torch.Tensor): Cumulative product of alphas for each timestep.
        betas (torch.Tensor): Array of beta values for each timestep.
        device (torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: The target noisy state.
    """
    num_dims = len(state.shape[1:])
    alpha_t, beta_t = get_alpha_beta_t(t, alpha_cumprod, betas, device, num_dims)
    return add_noise(state, alpha_t, beta_t, device)

def generate_noisy_state(state, t, alpha_cumprod, betas, device):
    """
    Generate the noisy state by adding noise at timestep t+1.

    Args:
        state (torch.Tensor): The original state.
        t (torch.Tensor): Timesteps tensor.
        alpha_cumprod (torch.Tensor): Cumulative product of alphas for each timestep.
        betas (torch.Tensor): Array of beta values for each timestep.
        device (torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: The noisy state.
    """
    num_dims = len(state.shape[1:])
    next_t = torch.clamp(t + 1, max=len(betas) - 1)
    alpha_t_plus_1, beta_t_plus_1 = get_alpha_beta_t(next_t, alpha_cumprod, betas, device, num_dims)
    return add_noise(state, alpha_t_plus_1, beta_t_plus_1, device)