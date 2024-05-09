"""
Custom GPT-2 Model Configuration

This module defines a custom implementation of the GPT-2 model using the Hugging Face `transformers` library.
It enables the customization of various GPT-2 configuration parameters such as the number of layers,
model dimensionality, number of attention heads, and dropout rates. This module is designed to provide
flexibility in constructing GPT-2 models tailored for specific tasks by adjusting its core architectural components.

Components:
- _GPTBase: A base class that initializes a GPT-2 model with a configurable architecture. The class
  uses the GPT2Config from the `transformers` library to set up the model according to the specified parameters.
- GPT: A derived class that extends _GPTBase, facilitating parameter initialization and providing a forward
  method that integrates seamlessly with PyTorch pipelines.

This tailored configuration is particularly useful for research and applications where model tuning is
critical for performance optimization.

References:
- Radford, A., et al. "Language Models are Unsupervised Multitask Learners." OpenAI Blog (2019).
- Wolf, T., et al. "Huggingface's Transformers: State-of-the-art Natural Language Processing."
  ArXiv, abs/1910.03771 (2020).
"""

import torch.nn as nn
from transformers import GPT2Model, GPT2Config

class _GPTBase(nn.Module):
    def __init__(self, num_layer, d_model, num_heads, dropout):
        super(_GPTBase, self).__init__()   
        config = GPT2Config(
            vocab_size=d_model,  # This should be set to your actual vocab size
            n_embd=d_model,
            n_layer=num_layer,
            n_head=num_heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False,
        )
        self.net = GPT2Model(config)

class GPT(_GPTBase):
    def __init__(self, network_params, num_heads: int = 8):
        num_layers = network_params.num_layers
        d_model = network_params.d_model
        dropout = network_params.dropout
        super(GPT, self).__init__(num_layers, d_model, num_heads, dropout)

    def forward(self, input_tensor, padding_mask=None):
        attention_mask = padding_mask.long() if padding_mask is not None else None
        output = self.net(inputs_embeds=input_tensor, attention_mask=attention_mask)
        output_tensor = output.last_hidden_state
        return output_tensor
