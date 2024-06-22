import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from nn.utils.init import init_weights
from nn.utils.transform_layer import TransformLayer

class ResNet(nn.Module):
    def __init__(self, network_config, pretrained_model):
        super(ResNet, self).__init__()
        d_model = network_config.d_model
        num_layers = network_config.num_layers
        num_channels, height, width = network_config.input_shape
        
        # Replace the initial conv1 layer if num_channels do not match
        if num_channels != pretrained_model.conv1.in_channels:
            pretrained_model.conv1 = nn.Conv2d(num_channels, 
                                     pretrained_model.conv1.out_channels, 
                                     kernel_size=pretrained_model.conv1.kernel_size, 
                                     stride=pretrained_model.conv1.stride, 
                                     padding=pretrained_model.conv1.padding, 
                                     bias=pretrained_model.conv1.bias)
            
        # Remove layers if height or width is less than required size
        min_sizes = [64, 32, 16]
        layers = ['layer4', 'layer3', 'layer2']
        num_ftrs = pretrained_model.fc.in_features
        
        # Loop through each layer and its corresponding min size
        for idx, (min_size, layer) in enumerate(zip(min_sizes, layers)):
            # Check if the layer should be removed based on input dimensions or the number of allowed layers
            if height < min_size or width < min_size or num_layers < 4 - idx:
                setattr(pretrained_model, layer, nn.Identity())  # Replace the layer with an identity layer to effectively remove it
                num_ftrs //= 2  # Halve the number of input features to the next layer or function
        
        pretrained_model.fc = nn.Linear(num_ftrs, d_model)  # Replace it with a new fc layer with d_model output features
        self.final_layer = TransformLayer(d_model, network_config.output_shape, first_act_fn='relu', last_act_fn=network_config.act_fn)
        
        self.pretrained_model = pretrained_model
        
        if num_channels != pretrained_model.conv1.in_channels:
            init_weights(self.pretrained_model.conv1)
        init_weights(self.pretrained_model.fc)

    def forward(self, x, padding_mask=None):
        x = self.pretrained_model(x)
        return self.final_layer(x)
    
def ResNet18(network_config):
    return ResNet(network_config, models.resnet18(pretrained=True))

def ResNet34(network_config):
    return ResNet(network_config, models.resnet34(pretrained=True))

def ResNet50(network_config):
    return ResNet(network_config, models.resnet50(pretrained=True))

class TransposeResnet(nn.Module):
    def __init__(self, network_config):
        super(TransposeResnet, self).__init__()

        try:
            import segmentation_models_pytorch as smp
            from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock
        except ImportError:
            raise ImportError("Error: segmentation_models_pytorch library is not installed. Please install it using 'pip install segmentation-models-pytorch'.")

        self.d_model = network_config.d_model
        self.num_channels, self.height, self.width = network_config.output_shape

        # Calculate the maximum number of layers based on the image size
        max_layers_height = math.ceil(math.log2(self.height))
        max_layers_width = math.ceil(math.log2(self.width))
        max_layers = min(max_layers_height, max_layers_width)

        # Ensure the number of layers does not exceed the maximum possible layers
        num_layers = min(network_config.num_layers, max_layers)

        self.initial_w = max(math.ceil(self.width / 2**num_layers), 1)
        self.initial_h = max(math.ceil(self.height / 2**num_layers), 1)
        
        minimum_channel_size = 16
        initial_channel_size = max(self.d_model, minimum_channel_size * (2 ** (num_layers - 1)))
        
        in_channels = [max(initial_channel_size // (2 ** i), minimum_channel_size) for i in range(num_layers)]
        out_channels = in_channels[1:] + [max(in_channels[-1] // 2, minimum_channel_size)]
        in_channels[0] = self.d_model
        
        skip_channels = [0 for _ in range(num_layers)]

        # Combine decoder keyword arguments
        kwargs = dict(use_batchnorm=True, attention_type=None)
        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ])

        # Define the final layer (Segmentation Head equivalent)
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels[-1],  # output channels of the last decoder block
                out_channels=self.num_channels,
                kernel_size=1
            ),
            nn.Tanh()  # Using Tanh activation function
        )

    def forward(self, x, padding_mask=None):
        # Ensure input x has the correct shape
        x = x.view(x.size(0), -1, 1, 1).repeat(1, 1, self.initial_h, self.initial_w)

        for decoder_block in self.blocks:
            skip = None
            x = decoder_block(x, skip)

        if x.size(2) != self.height or x.size(3) != self.width:
            x = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)

        # Apply the final layer
        x = self.final_layer(x)

        return x
