import torch.nn as nn
from structure.encoder import Encoder
from structure.decoder import Decoder
from structure.tokenizer import SentenceTokenizer

class Transformer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, vocab_size, num_layers=6, drop_prob=0.1):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, y):
        x = self.encoder(x)
        out = self.decoder(x, y)
        out = self.linear(out)
        return out