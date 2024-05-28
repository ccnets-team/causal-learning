# Check and import pytorch_tabnet if installed, else show an error message
try:
    from pytorch_tabnet.tab_network import TabNetEncoder as TabNetwork
except ImportError:
    print("Error: pytorch-tabnet library is not installed. Please install it using 'pip install pytorch-tabnet'.")
    raise

import torch
import torch.nn as nn

class TabNet(nn.Module):
    def __init__(self, network_params):
        super(TabNet, self).__init__()
        d_model = network_params.d_model
        num_layers = network_params.num_layers
        
        # Initialize group attention matrix with ones and move to appropriate device
        group_attention_matrix = torch.ones(1, d_model).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.network = TabNetwork(
            input_dim=d_model,
            output_dim=d_model,
            n_d = d_model,
            n_a = d_model,
            n_steps = num_layers,
            group_attention_matrix = group_attention_matrix
        )

    def forward(self, x, padding_mask=None):
        steps_output, _ = self.network(x)
        # steps_output is a list of tensors
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        return res