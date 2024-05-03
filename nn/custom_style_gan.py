'''
########################################################
Custom StyleGAN2 conditional generator and discriminator
########################################################
'''
import torch
import torch.nn as nn

class ConditionStyleMod(nn.Module):
    def __init__(self, latent_dim, condition_dim, channels):
        super().__init__()
        self.lin = nn.Linear(latent_dim + condition_dim, channels * 2)
    
    def forward(self, x, latent, condition):
        style = self.lin(torch.cat([latent, condition], dim=1))
        scale = style[:, :x.size(1)].unsqueeze(2).unsqueeze(3)
        shift = style[:, x.size(1):].unsqueeze(2).unsqueeze(3)
        return x * (scale + 1) + shift

class ConditionConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim, condition_dim, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.style_mod = ConditionStyleMod(latent_dim, condition_dim, out_channels)
        self.noise = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x, latent, condition):
        x = self.conv(x)
        x = x + self.noise * torch.randn_like(x)
        x = self.act(x)
        x = self.style_mod(x, latent, condition)
        return x
    
class ConditionalGenerator(nn.Module):
    def __init__(self, network_params):
        super().__init__()
        condition_dim = network_params.condition_dim
        z_dim = network_params.z_dim
        d_model = network_params.d_model
        num_layers = network_params.num_layers

        self.initial = nn.Parameter(torch.randn(1, d_model, 4, 4))  # Adjust initial layer
        self.style1 = ConditionStyleMod(z_dim, condition_dim, d_model)

        # Dynamically create convolution blocks
        blocks = []
        current_d_model = d_model
        for i in range(num_layers):
            next_d_model = max(1, d_model // (2 ** (i + 1)))  # Ensure at least 1 channel
            blocks.append(ConditionConvBlock(current_d_model, next_d_model, z_dim, condition_dim, 3, 1))
            current_d_model = next_d_model
        
        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(current_d_model, 3, 1),  # Use the output of the last block
            nn.Tanh(),
        )

    def forward(self, condition, style):
        batch_size = style.shape[0]
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
        z_dim = network_params.z_dim
        d_model = network_params.d_model
        num_layers = network_params.num_layers

        self.image_width = obs_shape[1]
        self.image_height = obs_shape[2]    
        self.condition_layer = nn.Linear(z_dim, int(self.image_width * self.image_height))

        # Setting up dynamic layer construction
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

    def forward(self, img, y):
        batch_size = img.size(0)
        condition = self.condition_layer(y).view(batch_size, 1, self.image_width, self.image_height)
        x = torch.cat([img, condition], dim=1)
        return self.main(x)
