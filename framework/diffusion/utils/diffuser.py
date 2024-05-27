
try:
    from framework.diffusion.utils.test import diffuse_data
except ImportError:
    import torch
    from framework.diffusion.utils.diffusion_utils import generate_target_state, generate_noisy_state

    def diffuse_data(state, condition, T, betas, alpha_cumprod, task_type):
        """
        Fallback diffuse_data implementation for testing purposes.

        Args:
            state (torch.Tensor): The original state.
            condition (torch.Tensor): The condition for diffusion.
            T (int): Number of timesteps.
            betas (torch.Tensor): Array of beta values for each timestep.
            alpha_cumprod (torch.Tensor): Cumulative product of alphas for each timestep.
            task_type (str): Type of task for diffusion.

        Returns:
            tuple: Noisy input state, target state, and input condition.
        """
        device = state.device
        batch_size = state.shape[0]
        
        t = torch.randint(0, T, (batch_size,), device=device)
        target_state = generate_target_state(state, t, alpha_cumprod, betas, device)
        input_state = generate_noisy_state(state, t, alpha_cumprod, betas, device)

        return input_state, target_state, condition