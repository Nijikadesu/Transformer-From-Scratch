import torch
import torch.nn as nn
import torch.nn.functional as F
from mechanism.layer_normalization import LayerNormalization
from mechanism.multihead_attention import MultiheadAttention
from mechanism.positional_encoding import PositionalEncoding
from mechanism.feed_forward import PositionwiseFeedForward
from mechanism.multihead_cross_attention import MultiheadCrossAttention

class DecoderLayer(nn.Module):
    """
    Implementation of Single Decoder Layer.
    """
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob=0.1):
        super().__init__()
        self.self_attention = MultiheadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.cross_attention = MultiheadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_mask=None, cross_mask=None):
        residual_y = y
        y = self.self_attention(y, mask=self_mask)
        y = self.dropout1(y)
        y = self.norm1(y + residual_y)

        residual_y = y
        y = self.cross_attention(x, y, mask=cross_mask)
        y = self.dropout2(y)
        y = self.norm2(y + residual_y)

        residual_y = y
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.norm3(y + residual_y)

        return y

class SequentialDecoder(nn.Sequential):
    """
    I tried to use nn.Sequential instead, but it didn't work out cuz the output of a single Sequential module is not
    complete, the next module not only requires the previous output y as input, but also x and mask, which is not given.

    This SequentialDecoder Provides a Complete Data Flow of Transformer Decoder.
    """
    def forward(self, *inputs):
        x, y, self_mask, cross_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_mask, cross_mask)
        return y

class Decoder(nn.Module):
    """
    Implementation of Transformer Decoder Architecture, including 6 Decoder Layers.
    """
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob=0.1, num_layers=6):
        super().__init__()
        self.layers = SequentialDecoder(*[DecoderLayer(d_model=d_model,
                                                   ffn_hidden=ffn_hidden,
                                                   num_heads=num_heads,
                                                   drop_prob=drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_mask=None, cross_mask=None):
        y = self.layers(x, y, self_mask, cross_mask)
        return y