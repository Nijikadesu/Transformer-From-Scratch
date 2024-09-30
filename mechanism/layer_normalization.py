import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """
    Implementation of Layer Normalization Mechanism.
    """
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps # eps is introduced to prevent the denominator std being zero
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # learnable scale parameter
        self.beta = nn.Parameter(torch.zeros(parameters_shape)) # learnable bias parameter

    def forward(self, input):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = input.mean(dim=dims, keepdim=True)
        var = (input - mean).pow(2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (input - mean) / std
        out = self.gamma * y + self.beta
        return out