import torch.nn as nn
from structure.encoder import Encoder
from structure.decoder import Decoder
from data_utils.tokenizer import SentenceTokenizer

class Transformer(nn.Module):
    """
    Implementation of the Transformer Neaural Network.
    In this file we combine the structure we build before and create the whole Transformer from scratch!
    Nice work!
    """
    def __init__(self, cfg):
        super().__init__()

        d_model = cfg.d_model
        ffn_hidden = cfg.ffn_hidden
        drop_prob = cfg.drop_prob
        num_heads = cfg.num_heads
        num_layers = cfg.num_layers
        vocab_size = cfg.vocab_size

        self.source_tokenizer = SentenceTokenizer(cfg, 'source')
        self.target_tokenizer = SentenceTokenizer(cfg, 'target')
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, y, device,
                encoder_self_attention_mask=None,
                decoder_self_attention_mask=None,
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False,
                dec_end_token=False):
        x = self.source_tokenizer(x, start_token=enc_start_token, end_token=enc_end_token) # tokenizing x
        y = self.target_tokenizer(y, start_token=dec_start_token, end_token=dec_end_token) # tokenizing y
        x = self.encoder(x, mask=encoder_self_attention_mask)
        out = self.decoder(x, y, self_mask=decoder_self_attention_mask, cross_mask=decoder_cross_attention_mask)
        out = self.linear(out)
        return out