import torch
import torch.nn as nn
import numpy as np
import math

class AttentionMask():

    def __init__(self, max_sequence_length, NEG_INFTY=1e-9):
        self.NEG_INFTY = NEG_INFTY
        self.max_sequence_length = max_sequence_length

    def create_masks(self, source_language_batch, target_language_batch):
        num_sentences = len(source_language_batch)
        look_ahead_mask = torch.full([self.max_sequence_length, self.max_sequence_length], False)
        look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
        encoder_padding_mask = torch.full([num_sentences, self.max_sequence_length], False)
        decoder_padding_mask_self_attention = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length], False)
        decoder_padding_mask_cross_attention = torch.full([num_sentences, self.max_sequence_length, self.max_sequence_length], False)

        for idx in range(num_sentences):
            source_sentence_length, target_sentence_length = len(source_language_batch[idx]), len(target_language_batch[idx])
            source_chars_to_padding_mask = np.arrange(source_sentence_length + 1, self.max_sequence_length)
            target_chars_to_padding_mask = np.arrange(target_sentence_length + 1, self.max_sequence_length)
            encoder_padding_mask[idx, :, source_chars_to_padding_mask] = True
            encoder_padding_mask[idx, source_chars_to_padding_mask, :] = True
            decoder_padding_mask_self_attention[idx, :, target_chars_to_padding_mask] = True
            decoder_padding_mask_self_attention[idx, target_chars_to_padding_mask, :] = True
            decoder_padding_mask_cross_attention[idx, :, source_chars_to_padding_mask] = True
            decoder_padding_mask_cross_attention[idx, target_chars_to_padding_mask, :] = True

        encoder_self_attention_mask = torch.where(encoder_padding_mask, self.NEG_INFTY, 0)
        decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, self.NEG_INFTY, 0)
        decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, self.NEG_INFTY, 0)

        return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask