import torch
import torch.nn as nn
import torch.nn.functional as F
from mechanism.layer_normalization import LayerNormalization
from mechanism.multihead_attention import MultiheadAttention
from mechanism.positional_encoding import PositionalEncoding
from mechanism.feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
    Implementation of Single Encoder Layer.
    """
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob=0.1):
        super().__init__()
        self.attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, mask=None):
        residual_x = x
        x = self.attention(x, mask=mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)

        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

class Encoder(nn.Module):
    """
    Implementation of Transformer Encoder Architecture, Including 6 Encoder Layers.
    """
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob=0.1, num_layers=6):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model=d_model,
                                                   ffn_hidden=ffn_hidden,
                                                   num_heads=num_heads,
                                                   drop_prob=drop_prob) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.layers(x, mask=mask)
        return x