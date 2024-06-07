import torch.nn as nn
import torchvision.models as models
from nn.utils.init import init_weights

class ResNet(nn.Module):
    def __init__(self, network_params, pretrained_model):
        super(ResNet, self).__init__()
        d_model = network_params.d_model
        num_layers = network_params.num_layers
        num_channels, height, width = network_params.obs_shape
        
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
        self.pretrained_model = pretrained_model
        
        if num_channels != pretrained_model.conv1.in_channels:
            init_weights(self.pretrained_model.conv1)
        init_weights(self.pretrained_model.fc)

    def forward(self, x):
        x = self.pretrained_model(x)
        return x
    
def ResNet18(network_params):
    return ResNet(network_params, models.resnet18(pretrained=True))

def ResNet34(network_params):
    return ResNet(network_params, models.resnet34(pretrained=True))

def ResNet50(network_params):
    return ResNet(network_params, models.resnet50(pretrained=True))

