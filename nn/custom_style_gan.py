"""
Custom StyleGAN2 Conditional Generator and Discriminator

This module contains customized implementations of the StyleGAN2 architecture,
including a Mapping Network that maps latent vectors to style codes used
across the generator for dynamic and conditional style modulation.

StyleGAN2 was originally developed by researchers at NVIDIA, with significant contributions from:
- Tero Karras: Principal architect of the StyleGAN series.
- Samuli Laine: Co-developer of the StyleGAN architectures.
- Miika Aittala: Contributed to advancements in image quality and training techniques.
- Janne Hellsten: Worked on optimization and refinement of generative models.
- Jaakko Lehtinen: Provided theoretical insights and practical improvements to the GAN training process.
- Timo Aila: Co-authored innovations in neural rendering and generative adversarial networks.

Generative Adversarial Networks (GANs) were originally invented by Ian Goodfellow in 2014.
This groundbreaking work laid the foundation for subsequent developments in the field,
including StyleGAN and StyleGAN2.

References:
- "Analyzing and Improving the Image Quality of StyleGAN" (Karras et al., 2019)
- "Generative Adversarial Nets" (Goodfellow et al., 2014)

This implementation may include modifications to the original design to suit specific conditional generation tasks.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_activation(act_name):
    if act_name == 'leaky_relu':
        return nn.LeakyReLU(0.2)
    elif act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {act_name}")
    
class MappingNetwork(nn.Module):
    """Maps the latent vector to style codes with several fully connected layers."""
    def __init__(self, latent_dim, style_dim, num_layers, act='leaky_relu'):
        super().__init__()
        # Determine the activation function based on the string identifier
        activation = get_activation(act)

        # First layer handles dimensionality change if needed
        layers = [nn.Sequential(nn.Linear(latent_dim, style_dim), activation)]

        self.mapping_layers = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.mapping_layers(z)

class StyleMod(nn.Module):
    """
    Applies a learned affine transformation based on style vectors.
    Optionally includes a condition vector for combined transformations.
    If no dimensions are provided, acts as a pass-through.
    """
    def __init__(self, channels, style_dim=None, condition_dim=None):
        super().__init__()
        self.channels = channels
        input_dim = (style_dim if style_dim is not None else 0) + (condition_dim if condition_dim is not None else 0)
        
        if input_dim > 0:
            self.lin = nn.Linear(input_dim, channels * 2)
        else:
            self.lin = None

    def forward(self, x, style=None, condition=None):
        if self.lin is not None:
            if condition is not None:
                style = torch.cat([style, condition], dim=1) if style is not None else condition
            style_transform = self.lin(style)
            scale = style_transform[:, :self.channels].unsqueeze(2).unsqueeze(3)
            shift = style_transform[:, self.channels:].unsqueeze(2).unsqueeze(3)
            x = x * scale + shift
        return x

class ConvolutionalBlock(nn.Module):
    """Unified convolutional block for noise, style, and conditional style modulation."""
    def __init__(self, in_channels, out_channels, use_noise=True, style_dim=None, condition_dim=None, act='leaky_relu'):
        super().__init__()
        self.use_noise = use_noise
        activation = get_activation(act)
        
        # Initialize the first convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.noise = nn.Parameter(torch.randn(1, out_channels, 1, 1)) if use_noise else None
        self.act = activation

        # Optionally apply style modulation
        self.style_mod = StyleMod(out_channels, style_dim, condition_dim)

        # Initialize the second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x, style=None, condition=None):
        x = self.conv1(x)
        if self.noise is not None:
            x += self.noise * torch.randn_like(x)
        x = self.act(x)
        x = self.style_mod(x, style, condition)

        x = self.conv2(x)
        x = self.act(x)
        
        return x

class Generator(nn.Module):
    """ Generator model incorporating a mapping network and conditionally applied styles. """
    def __init__(self, network_params):
        super().__init__()
        d_model = network_params.d_model
        num_channels, height, width = network_params.obs_shape
        num_layers = network_params.num_layers
        self.style_dim = d_model
        self.mapping_network = MappingNetwork(self.style_dim, d_model, num_layers=num_layers, act='relu')
        self.style1 = StyleMod(channels=d_model, style_dim=self.style_dim)
        self.blocks = nn.ModuleList()
        current_d_model = d_model
        
        for i in range(num_layers):
            next_d_model = max(num_channels, current_d_model // 2)  # Ensure not below num_channels
            self.blocks.append(ConvolutionalBlock(current_d_model, next_d_model, use_noise=True, style_dim=self.style_dim, act='relu'))
            current_d_model = next_d_model
        
        self.to_rgb = nn.Sequential(nn.Conv2d(current_d_model, num_channels, 1), nn.Tanh())
        self.height = height
        self.width = width

    def forward(self, z, padding_mask=None):
        style = self.mapping_network(z)
        batch_size = z.shape[0]
        # Start with a small spatial dimension that will be scaled up to the desired size
        out = z.view(batch_size, -1, 1, 1).repeat(1, 1, 2, 2)
        out = self.style1(out, style)
        
        for block in self.blocks:
            if out.size(2) < self.height and out.size(3) < self.width: 
                out = F.interpolate(out, scale_factor=2, mode='nearest')
            out = block(out, style)
        
        # Ensure the output has the exact dimensions required (using adaptive average pooling to adjust final size)
        if out.size(2) != self.height or out.size(3) != self.width:
            out = F.interpolate(out, size=(self.height, self.width), mode='bilinear', align_corners=False)
        out = self.to_rgb(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, network_params):
        super().__init__()
        self.z_dim = network_params.z_dim
        self.d_model = network_params.d_model
        num_channels, height, width = network_params.obs_shape
        num_layers = network_params.num_layers

        self.blocks = nn.ModuleList()
        in_channels = num_channels  # Starting with RGB channels
        
        for i in range(num_layers):
            out_channels = min(self.d_model, self.d_model // (2 ** (num_layers - i - 1)))
            self.blocks.append(ConvolutionalBlock(in_channels, out_channels, use_noise=False, act = 'relu'))
            in_channels = out_channels

        final = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, self.d_model)]
        self.final = nn.Sequential(*final)

    def forward(self, img, padding_mask=None):
        x = img
        for block in self.blocks:
            x = block(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return self.final(x)
    
class ConditionalDiscriminator(nn.Module):
    def __init__(self, network_params):
        super().__init__()
        self.z_dim = network_params.z_dim
        self.d_model = network_params.d_model
        self.style_dim = self.d_model  # Ensure style_dim is defined correctly
        num_channels, height, width = network_params.obs_shape
        num_layers = network_params.num_layers
        
        self.mapping_network = MappingNetwork(self.z_dim, self.style_dim, num_layers, act = 'relu')

        self.blocks = nn.ModuleList()
        in_channels = num_channels  # Starting with RGB channels
        
        for i in range(num_layers):
            out_channels = min(self.d_model, self.d_model // (2 ** (num_layers - i - 1)))
            self.blocks.append(ConvolutionalBlock(in_channels, out_channels, use_noise=False, style_dim = self.style_dim, act = 'relu'))
            in_channels = out_channels

        final = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(out_channels, self.d_model)]
        self.final = nn.Sequential(*final)

    def forward(self, img, e, padding_mask=None):
        x = img
        # Generate style codes from the condition vector
        style = self.mapping_network(e)
        
        for block in self.blocks:
            x = block(x, style)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        return self.final(x)