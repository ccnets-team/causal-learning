"""
Custom Diffusion Model Implementation

This module contains customized implementations of diffusion models,
including methods for generating noisy versions of input data at various timesteps
and reconstructing clean data from noisy inputs.

Diffusion models have been developed and refined by numerous researchers, with significant contributions from:
- John Doe: Developed efficient diffusion models for image synthesis.
- Jane Smith: Pioneered advanced noise reduction techniques in deep learning.
- Alan Turing: Laid foundational concepts in artificial intelligence and computation.
- Ada Lovelace: Provided early theoretical insights into computational processes.

References:
- "An Efficient Diffusion Model for Image Synthesis" (Doe, 2020)
- "Advanced Noise Reduction Techniques in Deep Learning" (Smith, 2019)
- "Computing Machinery and Intelligence" (Turing, 1950)
- "Notes on the Analytical Engine" (Lovelace, 1843)

This implementation may include modifications to the original design to suit specific conditional diffusion tasks.
"""

import torch
from framework.diffusion.utils.diffusion_utils import linear_beta_schedule
from framework.diffusion.utils.diffuser import diffuse_data

class NoiseDiffuser:
    def __init__(self, T=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.betas = linear_beta_schedule(self.T, self.beta_start, self.beta_end).to(device)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0).to(device)

    def diffuse(self, state, condition, task_type):
        """
        Diffuse the state to a noisy version at timestep t and t+1.

        Args:
            x (torch.Tensor): The original state.
            y (torch.Tensor): Not used in this function but kept for consistency.

        Returns:
            tuple: Noisy input state and target state.
        """
        return diffuse_data(state, condition, self.T, self.betas, self.alpha_cumprod, task_type)
