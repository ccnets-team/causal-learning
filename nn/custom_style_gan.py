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
import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    """ Maps the latent vector to style codes with several fully connected layers. """
    def __init__(self, latent_dim, style_dim, num_layers):
        super().__init__()
        layers = [nn.Sequential(nn.Linear(latent_dim, style_dim), nn.LeakyReLU(0.2)) for _ in range(num_layers)]
        self.mapping_layers = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.mapping_layers(z)

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

class ConditionConvBlock(nn.Module):
    """ Convolutional block applying style and noise modulation. """
    def __init__(self, in_channels, out_channels, style_dim, condition_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.style_mod = ConditionStyleMod(style_dim, condition_dim, out_channels)
        self.noise = nn.Parameter(torch.randn(1, out_channels, 1, 1))
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x, style, condition):
        x = self.conv(x)
        x = x + self.noise * torch.randn_like(x)
        x = self.act(x)
        x = self.style_mod(x, style, condition)
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
        self.initial = nn.Parameter(torch.randn(1, d_model, 2, 2))
        self.mapping_network = MappingNetwork(z_dim, self.style_dim, num_layers)
        self.style1 = ConditionStyleMod(self.style_dim, condition_dim, d_model)
        
        self.blocks = nn.ModuleList()
        current_d_model = d_model
        for i in range(num_layers):
            next_d_model = max(1, current_d_model // 2)  # Reduce channel count
            self.blocks.append(ConditionConvBlock(current_d_model, next_d_model, self.style_dim, condition_dim))
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
        d_model = network_params.d_model
        num_layers = network_params.num_layers
        
        layers = []
        in_channels = 3
        for i in range(num_layers):
            out_channels = d_model // (2 ** (num_layers - i - 1))
            stride = 1 if i == 0 else 2  # First layer does not reduce spatial dimensions.
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels, d_model))

        self.main = nn.Sequential(*layers)

    def forward(self, img):
        return self.main(img)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, network_params):
        super().__init__()
        obs_shape = network_params.obs_shape
        self.z_dim = network_params.z_dim
        d_model = network_params.d_model
        num_layers = network_params.num_layers

        self.image_width = obs_shape[1]
        self.image_height = obs_shape[2]    
        self.image_elements = torch.prod(torch.tensor(obs_shape[1:], dtype=torch.int)).item()
        self.obs_shape = obs_shape
        
        # Initialize layers for dynamic construction
        layers = []
        in_channels = 4  # Starting from 3 + 1 for RGB channels and the additional condition channel
        for i in range(num_layers):
            out_channels = d_model // (2 ** (num_layers - i - 1))
            stride = 2 if i > 0 else 1  # The first layer has a stride of 1 to maintain more spatial information
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels, d_model))

        self.main = nn.Sequential(*layers)

    def _convert_explanation_to_image_shape(self, e):
        """ Convert the explanation vector to match the target image shape with the first dimension set to 1. """
        explain_shape = [1] + list(self.obs_shape[1:])  # Set first dim to 1, rest match target shape
        e1 = e.repeat(1, self.image_elements // self.z_dim)
        e2 = torch.zeros_like(e[:, :self.image_elements % self.z_dim])
        expanded_e = torch.cat([e1, e2], dim=-1)  # Repeat to match the volume of target shape
        expanded_e = expanded_e.view(-1, *explain_shape)  # Reshape explanation vector to the new explain_shape
        return expanded_e

    def forward(self, img, y):
        # Convert the explanation vector to match the target image shape
        condition = self._convert_explanation_to_image_shape(y)
        
        # Concatenate the image and condition along the channel dimension
        x = torch.cat([img, condition], dim=1)
        
        return self.main(x)
