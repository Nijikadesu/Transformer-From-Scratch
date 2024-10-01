import torch
import numpy as np
import torch.nn as nn
from mechanism.positional_encoding import PositionalEncoding

class SentenceTokenizer(nn.Module):

    def __init__(self, cfg, language='target'):
        super().__init__()
        language_to_index = cfg.source_to_index if language == 'source' else cfg.target_to_index
        self.vocab_size = len(language_to_index)
        self.max_sequence_size = cfg.max_sequence_size
        self.embedding = nn.Embedding(self.vocab_size, cfg.d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(cfg.d_model, cfg.max_sequence_size)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = cfg.START_TOKEN
        self.END_TOKEN = cfg.END_TOKEN
        self.PADDING_TOKEN = cfg.PADDING_TOKEN

    def batch_tokenize(self, batch, start_token=True, end_token=True):

        def tokenize(sentence, start_token=True, end_token=True):
            sentence_word_indices = [self.language_to_index[token] for token in list(sentence)]
            if start_token:
                sentence_word_indices.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indices.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indices), self.max_sequence_length):
                sentence_word_indices.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indices)

        tokenized = []
        for sentence_id in range(len(batch)):
            tokenized.append(tokenize(batch[sentence_id], start_token, end_token))
        tokenized = torch.stack(tokenized)
        return tokenized

    def forward(self, x, start_token=True, end_token=True):
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder()
        x = self.dropout(x + pos)
        return x