
import torch
import torch.nn as nn
from nn.utils.transform_layer import TransformLayer

class EncoderTabNet(nn.Module):
    def __init__(self, network_config):
        # Check and import pytorch_tabnet if installed, else show an error message
        try:
            from pytorch_tabnet.tab_network import TabNetEncoder
        except ImportError:
            print("Error: pytorch-tabnet library is not installed. Please install it using 'pip install pytorch-tabnet'.")
            raise
        
        super(EncoderTabNet, self).__init__()
        d_model = network_config.d_model
        num_layers = network_config.num_layers
        output_size = network_config.output_shape[-1]
        
        # Initialize group attention matrix with ones and move to appropriate device
        group_attention_matrix = torch.ones(1, d_model).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.network = TabNetEncoder(
            input_dim=d_model,
            output_dim=d_model,
            n_d = d_model,
            n_a = d_model,
            n_steps = num_layers,
            group_attention_matrix = group_attention_matrix
        )
        self.final_layer = TransformLayer(d_model, output_size, first_act_fn='relu', last_act_fn=network_config.act_fn)

    def forward(self, x, padding_mask=None):
        steps_output, _ = self.network(x)
        # steps_output is a list of tensors
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        
        return self.final_layer(res)
    
class DecoderTabNet(nn.Module):
    def __init__(self, network_config):
        
        # Check if PyTorch-TabNet is installed
        try:
            from pytorch_tabnet.tab_network import TabNetDecoder
        except ImportError:
            raise ImportError("pytorch-tabnet library is not installed. Please install it using 'pip install pytorch-tabnet'.")

        super(DecoderTabNet, self).__init__()
        
        d_model = network_config.d_model
        num_layers = network_config.num_layers
        output_size = network_config.output_shape[-1]
        
        self.decoder = TabNetDecoder(
            input_dim=d_model,
            n_d=d_model,
            n_steps=num_layers
        )
        self.num_layers = num_layers
        self.final_layer = TransformLayer(d_model, output_size, first_act_fn='relu', last_act_fn=network_config.act_fn)

    def forward(self, x, padding_mask=None):
        res = []
        for i in range(self.num_layers):
            res.append(x.unsqueeze(0))  # Add dimension [1, batch_size, features]
        
        res = torch.cat(res, dim=0)  # [num_layers, batch_size, split_size]

        # Forward pass through the decoder
        reconstructed_input = self.decoder(res)
        return self.final_layer(reconstructed_input)