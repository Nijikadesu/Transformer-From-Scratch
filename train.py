import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer import Transformer
from config import Config
from dataset import TextDataset
from mechanism.attention_mask import AttentionMask

class Trainer():
    def __init__(self, cfg):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Transformer(cfg).to(self.device)
        self.dataloader = DataLoader(TextDataset(cfg), batch_size=cfg.batch_size, shuffle=False)
        # let optimizer ignore cases when the label is a padding token
        self.criterion = nn.CrossEntropyLoss(ignore_index=cfg.target_to_index[cfg.PADDING_TOKEN], reduction='none')

        for params in self.model.parameters():
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.num_epochs = cfg.num_epochs

        self.mask = AttentionMask(cfg)

        self.enc_start_token = cfg.enc_start_token
        self.enc_end_token = cfg.enc_end_token
        self.dec_start_token = cfg.dec_start_token
        self.dec_end_token = cfg.dec_end_token

        self.vocab_size = cfg.vocab_size
        self.target_to_index = cfg.target_to_index
        self.PADDING_TOKEN = cfg.PADDING_TOKEN

    def train(self):
        self.model.train()
        total_loss = 0

        for epoch in range(self.num_epochs):
            print(f'[ Train Epoch {epoch+1:03d} / {self.num_epochs:03d}]')
            for idx, (x, y) in enumerate(self.dataloader):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = self.mask(x, y)
                pred = self.model(x, y, device=self.device,
                                  encoder_self_attention_mask=encoder_self_attention_mask,
                                  decoder_self_attention_mask=decoder_self_attention_mask,
                                  decoder_cross_attention_mask=decoder_cross_attention_mask,
                                  enc_start_token=self.enc_start_token,
                                  enc_end_token=self.enc_end_token,
                                  dec_start_token=self.dec_start_token,
                                  dec_end_token=self.dec_end_token)
                labels = self.model.target_tokenizer.batch_tokenize(y, start_token=self.dec_start_token, end_token=self.dec_end_token)
                loss = self.criterion(
                    pred.view(-1, self.vocab_size).to(self.device),
                    labels.view(-1).to(self.device)
                ).to(self.device)
                # set padding position to False
                valid_indices = torch.where(labels.view(-1) == self.target_to_index[self.PADDING_TOKEN], False, True)
                loss = loss.sum() / valid_indices.sum()
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                ### to be continue