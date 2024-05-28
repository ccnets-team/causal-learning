import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
    
class TransposeResnet(nn.Module):
    def __init__(self, network_params, encoder_name):
        super(TransposeResnet, self).__init__()
        self.d_model = network_params.d_model
        self.num_channels, self.height, self.width = network_params.obs_shape

        # Initialize a pretrained UNet model from the segmentation_models_pytorch library
        unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",            
            in_channels=self.d_model,      
            classes=self.num_channels       # Output channels to match the final number of channels
        )

        # # Extract the decoder part of the U-Net
        self.decoder = unet.decoder

        # Obtain the number of output channels from the last decoder block
        
        # Define the final layer (Segmentation Head equivalent)
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=unet.segmentation_head[0].in_channels + self.d_model, 
                out_channels=self.num_channels, 
                kernel_size=3, 
                padding=1
            ),
            nn.Tanh()  # Using Tanh activation function
        )

    def forward(self, x, padding_mask=None):
        # Pass x through the decoder
        x = x.view(x.size(0), -1, 1, 1).repeat(1, 1, 4, 4)
        x = [None] + [x]
        x = self.decoder(*x)
        # Apply the final layer
        x = self.final_layer(x)
        return x
    
def TransposeResNet18(network_params):
    return TransposeResnet(network_params, 'resnet18')

def TransposeResNet34(network_params):
    return TransposeResnet(network_params, 'resnet34')

def TransposeResNet50(network_params):
    return TransposeResnet(network_params, 'resnet50')
