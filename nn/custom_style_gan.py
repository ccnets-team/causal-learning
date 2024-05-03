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
    def __init__(self, in_channels, out_channels, latent_dim, condtion_dim, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.style1 = ConditionStyleMod(latent_dim, condtion_dim, out_channels)
        self.style2 = ConditionStyleMod(latent_dim, condtion_dim, out_channels)
        self.noise1 = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.noise2 = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.act = nn.LeakyReLU(0.2)
    def forward(self, x, latent, condition):
        x = self.conv1(x)
        rand1 = torch.randn_like(x)
        x = x + self.noise1 * rand1
        x = self.act(x)
        x = self.style1(x, latent, condition)
        x = self.conv2(x)
        rand2 = torch.randn_like(x)
        x = x + self.noise2 * rand2
        x = self.act(x)
        x = self.style2(x, latent, condition)
        return x
            
class ConditionalGenerator(nn.Module):
    def __init__(self, network_params):
        super().__init__()
        condition_dim = network_params.condition_dim
        z_dim = network_params.z_dim
        channel_multiplier = network_params.channel_multiplier
        self.initial = nn.Parameter(torch.randn(1, channel_multiplier, 4, 4))
        self.style1 = ConditionStyleMod(z_dim, condition_dim, channel_multiplier)
        self.blocks = nn.ModuleList([
            ConditionConvBlock(channel_multiplier, channel_multiplier, z_dim, condition_dim, 3, 1),
            ConditionConvBlock(channel_multiplier, channel_multiplier, z_dim, condition_dim, 3, 1),
            ConditionConvBlock(channel_multiplier, channel_multiplier, z_dim, condition_dim, 3, 1),
            ConditionConvBlock(channel_multiplier, channel_multiplier, z_dim, condition_dim, 3, 1),
            ConditionConvBlock(channel_multiplier, channel_multiplier, z_dim, condition_dim, 3, 1),
        ])
        self.to_rgb = nn.Sequential(
            nn.Conv2d(channel_multiplier, 3, 1),
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
        output_size = network_params.z_dim
        channel_multiplier = network_params.channel_multiplier
        self.main = nn.Sequential(
            nn.Conv2d(3, channel_multiplier, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_multiplier, channel_multiplier * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_multiplier * 2, channel_multiplier * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_multiplier * 4, channel_multiplier * 8, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel_multiplier * 8 * 1 * 1, output_size),
        )
    def forward(self, img):
        return self.main(img)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, network_params):
        super().__init__()
        output_size = network_params.condition_dim
        obs_shape = network_params.obs_shape
        z_dim = network_params.z_dim
        channel_multiplier = network_params.channel_multiplier
        self.image_width = obs_shape[1]
        self.image_height = obs_shape[2]    
        self.condition_layer = nn.Linear(z_dim, int(self.image_width * self.image_height))
        self.main = nn.Sequential(
            nn.Conv2d(3 + 1, channel_multiplier, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_multiplier, channel_multiplier * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_multiplier * 2, channel_multiplier * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channel_multiplier * 4, channel_multiplier * 8, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel_multiplier * 8 * 1 * 1, output_size),
        )
    def forward(self, img, y):
        batch_size = img.size(0)
        condition = self.condition_layer(y).view(batch_size, 1, self.image_width, self.image_height)
        x = torch.cat([img, condition], dim=1)
        return self.main(x)