########################################################
# Custom StyleGAN2 Conditional Generator and Discriminator
# This module contains customized implementations of the StyleGAN2 architecture,
# including a Mapping Network that maps latent vectors to style codes used
# across the generator for dynamic and conditional style modulation.
#
# StyleGAN2 was originally developed by researchers at NVIDIA, with significant contributions from:
# - Tero Karras: Principal architect of the StyleGAN series.
# - Samuli Laine: Co-developer of the StyleGAN architectures.
# - Miika Aittala: Contributed to advancements in image quality and training techniques.
# - Janne Hellsten: Worked on optimization and refinement of generative models.
# - Jaakko Lehtinen: Provided theoretical insights and practical improvements to the GAN training process.
# - Timo Aila: Co-authored innovations in neural rendering and generative adversarial networks.
#
# Generative Adversarial Networks (GANs) were originally invented by Ian Goodfellow in 2014.
# This groundbreaking work laid the foundation for subsequent developments in the field,
# including StyleGAN and StyleGAN2.
#
# References:
# "Analyzing and Improving the Image Quality of StyleGAN" (Karras et al., 2019)
# "Generative Adversarial Nets" (Goodfellow et al., 2014)
# This implementation may include modifications to the original design to suit specific conditional generation tasks.
########################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    """ Maps the latent vector to style codes with several fully connected layers. """
    def __init__(self, latent_dim, style_dim, num_layers):
        super().__init__()
        layers = [nn.Sequential(nn.Linear(latent_dim, style_dim), nn.LeakyReLU(0.2)) for _ in range(num_layers)]
        self.mapping_layers = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.mapping_layers(z)

class StyleMod(nn.Module):
    """Applies a learned affine transformation based on style vectors."""
    def __init__(self, style_dim, channels):
        super().__init__()
        self.channels = channels
        self.lin = nn.Linear(style_dim, channels * 2)

    def forward(self, x, style):
        style_transform = self.lin(style)
        scale = style_transform[:, :self.channels].unsqueeze(2).unsqueeze(3)
        shift = style_transform[:, self.channels:].unsqueeze(2).unsqueeze(3)
        return x * scale + shift
    
class ConditionStyleMod(nn.Module):
    """ Applies a learned affine transformation based on style and condition vectors. """
    def __init__(self, style_dim, condition_dim, channels):
        super().__init__()
        self.channels = channels
        self.lin = nn.Linear(style_dim + condition_dim, channels * 2)
    
    def forward(self, x, style, condition):
        combined_input = torch.cat([style, condition], dim=1)
        style = self.lin(combined_input)
        scale = style[:, :self.channels].unsqueeze(2).unsqueeze(3)
        shift = style[:, self.channels:].unsqueeze(2).unsqueeze(3)
        return x * scale + shift

class ConvBlock(nn.Module):
    """Convolutional block applying noise modulation."""
    def __init__(self, in_channels, out_channels, use_noise=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.use_noise = use_noise
        if use_noise:
            self.noise1 = nn.Parameter(torch.randn(1, out_channels, 1, 1))
            self.noise2 = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_noise:
            x = x + self.noise1 * torch.randn_like(x)
        x = self.act(x)

        x = self.conv2(x)
        if self.use_noise:
            x = x + self.noise2 * torch.randn_like(x)
        x = self.act(x)

        return x
        
class StyleConvBlock(nn.Module):
    """Convolutional block applying style and noise modulation."""
    def __init__(self, in_channels, out_channels, style_dim, use_noise=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.style_mod1 = StyleMod(style_dim, out_channels)
        self.style_mod2 = StyleMod(style_dim, out_channels)
        self.use_noise = use_noise
        if use_noise:
            self.noise1 = nn.Parameter(torch.randn(1, out_channels, 1, 1))
            self.noise2 = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, style = None):
        x = self.conv1(x)
        if self.use_noise:
            x = x + self.noise1 * torch.randn_like(x)
        x = self.act(x)
        if style is not None:
            x = self.style_mod1(x, style)

        x = self.conv2(x)
        if self.use_noise:
            x = x + self.noise2 * torch.randn_like(x)
        x = self.act(x)
        if style is not None:
            x = self.style_mod2(x, style)

        return x
    
class ConditionStyleConvBlock(nn.Module):
    """ Convolutional block applying style and noise modulation. """
    def __init__(self, in_channels, out_channels, style_dim, condition_dim, use_noise=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.style_mod1 = ConditionStyleMod(style_dim, condition_dim, out_channels)
        self.style_mod2 = ConditionStyleMod(style_dim, condition_dim, out_channels)

        if use_noise:
            self.noise1 = nn.Parameter(torch.randn(1, out_channels, 1, 1))
            self.noise2 = nn.Parameter(torch.randn(1, out_channels, 1, 1))

        self.act = nn.LeakyReLU(0.2)
        self.use_noise = use_noise
    
    def forward(self, x, style, condition):
        x = self.conv1(x)
        if self.use_noise:
            x = x + self.noise1 * torch.randn_like(x)
        x = self.act(x)
        x = self.style_mod1(x, style, condition)

        x = self.conv2(x)
        if self.use_noise:
            x = x + self.noise2 * torch.randn_like(x)
        x = self.act(x)
        x = self.style_mod2(x, style, condition)

        return x

class ConditionalGenerator(nn.Module):
    """ Generator model incorporating a mapping network and conditionally applied styles. """
    def __init__(self, network_params):
        super().__init__()
        condition_dim = network_params.condition_dim
        z_dim = network_params.z_dim
        d_model = network_params.d_model
        num_layers = network_params.num_layers
        
        self.style_dim = z_dim  
        self.initial = nn.Parameter(torch.randn(1, d_model, 4, 4))
        self.mapping_network = MappingNetwork(z_dim, self.style_dim, num_layers)
        self.style1 = ConditionStyleMod(self.style_dim, condition_dim, d_model)
        
        self.blocks = nn.ModuleList()
        current_d_model = d_model
        for i in range(num_layers):
            next_d_model = max(1, current_d_model // 2)  # Reduce channel count
            self.blocks.append(ConditionStyleConvBlock(current_d_model, next_d_model, self.style_dim, condition_dim, use_noise=True))
            current_d_model = next_d_model
        
        self.to_rgb = nn.Sequential(nn.Conv2d(current_d_model, 3, 1), nn.Tanh())

    def forward(self, z, condition):
        style = self.mapping_network(z)
        batch_size = z.shape[0]
        out = self.initial.expand(batch_size, -1, -1, -1)
        out = self.style1(out, style, condition)
        for block in self.blocks:
            out = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
            out = block(out, style, condition)
        out = self.to_rgb(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, network_params):
        super().__init__()
        self.z_dim = network_params.z_dim
        self.d_model = network_params.d_model
        num_layers = network_params.num_layers

        self.blocks = nn.ModuleList()
        in_channels = 3  # Starting with RGB channels
        
        for i in range(num_layers):
            out_channels = self.d_model // (2 ** (num_layers - i - 1))
            self.blocks.append(ConvBlock(in_channels, out_channels, use_noise=False))
            in_channels = out_channels

        final = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, self.d_model)]
        self.final = nn.Sequential(*final)

    def forward(self, img):
        x = img
        for block in self.blocks:
            x = block(x)
            # Apply down-sampling after processing with the block
            if x.size(2) > 1 and x.size(3) > 1:  # Avoid reducing too small dimensions
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return self.final(x)
    
class ConditionalDiscriminator(nn.Module):
    def __init__(self, network_params):
        super().__init__()
        self.z_dim = network_params.z_dim
        self.d_model = network_params.d_model
        self.style_dim = self.z_dim  # Ensure style_dim is defined correctly
        num_layers = network_params.num_layers
        self.mapping_network = MappingNetwork(self.z_dim, self.style_dim, num_layers)

        self.blocks = nn.ModuleList()
        in_channels = 3  # Starting with RGB channels
        
        for i in range(num_layers):
            out_channels = self.d_model // (2 ** (num_layers - i - 1))
            self.blocks.append(StyleConvBlock(in_channels, out_channels, self.style_dim, use_noise=False))
            in_channels = out_channels

        final = [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, self.d_model)]
        self.final = nn.Sequential(*final)

    def forward(self, img, e):
        x = img
        # Generate style codes from the condition vector
        style = self.mapping_network(e)
        
        for block in self.blocks:
            x = block(x, style)
            # Apply down-sampling after processing with the block
            if x.size(2) > 1 and x.size(3) > 1:  # Avoid reducing too small dimensions
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return self.final(x)