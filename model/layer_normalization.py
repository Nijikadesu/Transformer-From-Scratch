import torch
import torch.nn as nn

class LayerNormalization(nn.Module):

    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, input):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = input.mean(dim=dims, keepdim=True)
        var = (input - mean).pow(2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (input - mean) / std
        out = self.gamma * y + self.beta
        return out