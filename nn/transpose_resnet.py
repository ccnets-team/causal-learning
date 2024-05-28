import math
import torch.nn as nn
import torch.nn.functional as F

class TransposeResnet(nn.Module):
    def __init__(self, network_params):
        super(TransposeResnet, self).__init__()

        try:
            import segmentation_models_pytorch as smp
            from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock
        except ImportError:
            raise ImportError("Error: segmentation_models_pytorch library is not installed. Please install it using 'pip install segmentation-models-pytorch'.")

        self.d_model = network_params.d_model
        self.num_channels, self.height, self.width = network_params.obs_shape

        # Calculate the maximum number of layers based on the image size
        max_layers_height = math.ceil(math.log2(self.height))
        max_layers_width = math.ceil(math.log2(self.width))
        max_layers = min(max_layers_height, max_layers_width)

        # Ensure the number of layers does not exceed the maximum possible layers
        num_layers = min(network_params.num_layers, max_layers)

        self.initial_w = max(math.ceil(self.width / 2**num_layers), 1)
        self.initial_h = max(math.ceil(self.height / 2**num_layers), 1)
        
        initial_channel_size = 512
        minimum_channel_size = 32
        
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
            x = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=False)

        # Apply the final layer
        x = self.final_layer(x)

        return x
