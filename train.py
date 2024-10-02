import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer import Transformer
from config import Config
from dataset import TextDataset
from mechanism.attention_mask import AttentionMask

class Trainer():
    """
    We build a Trainer Class to integrade functions and variables we need to train a Transformer model.
    Because of the limit of my local computing resources, I set the Transformer layer to One.
    But you can modify it as well as other variables / hyper parameters in config.py.
    """
    def __init__(self, cfg):
        print("=" * 50)
        print('Initializing Trainer...')
        self.device = cfg.device
        self.model = Transformer(cfg).to(self.device)
        print('Finished Building Transformer with {} layer...'.format(cfg.num_layers))
        self.dataloader = DataLoader(TextDataset(cfg), batch_size=cfg.batch_size, shuffle=False)
        # let optimizer ignore cases when the label is a padding token
        self.criterion = nn.CrossEntropyLoss(ignore_index=cfg.target_to_index[cfg.PADDING_TOKEN], reduction='none')

        for params in self.model.parameters():
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.num_epochs = cfg.num_epochs

        self.mask = AttentionMask(cfg)

        self.vocab_size = cfg.vocab_size
        self.max_sequence_length = cfg.max_sequence_length
        self.target_to_index = cfg.target_to_index
        self.index_to_target = cfg.index_to_target

        self.START_TOKEN = cfg.START_TOKEN
        self.PADDING_TOKEN = cfg.PADDING_TOKEN
        self.END_TOKEN = cfg.END_TOKEN

        self.model_path = cfg.model_path

        print('Done!')
        print("=" * 50)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print('Saved model to {}'.format(self.model_path))
        pass

    def train(self):
        total_loss = 0
        print('\nTraining model on device {}\n'.format(self.device))
        for epoch in range(self.num_epochs):
            print("=" * 50)
            print(f'[ Train Epoch {epoch+1:03d} / {self.num_epochs:03d}]')
            print('-' * 50)
            for batch_idx, (x, y) in enumerate(self.dataloader):
                self.model.train()
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = self.mask.create_masks(x, y)
                encoder_self_attention_mask = encoder_self_attention_mask.to(self.device)
                decoder_self_attention_mask = decoder_self_attention_mask.to(self.device)
                decoder_cross_attention_mask = decoder_cross_attention_mask.to(self.device)
                pred = self.model(x, y, device=self.device,
                                  encoder_self_attention_mask=encoder_self_attention_mask,
                                  decoder_self_attention_mask=decoder_self_attention_mask,
                                  decoder_cross_attention_mask=decoder_cross_attention_mask,
                                  enc_start_token=False,
                                  enc_end_token=False,
                                  dec_start_token=True,
                                  dec_end_token=True)
                labels = self.model.target_tokenizer.batch_tokenize(y, start_token=True, end_token=True)
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

                if batch_idx % 100 == 0:
                    print(f'Iteration {batch_idx} | loss = {loss.item()}')
                    print(f'Source language: {x[0]}')
                    print(f'Target translation: {y[0]}')
                    sentence_pred = torch.argmax(pred[0], axis=1)
                    sentence = ""
                    for idx in sentence_pred:
                        if idx == self.target_to_index[self.START_TOKEN]:
                            continue
                        if idx == self.target_to_index[self.END_TOKEN]:
                            break
                        sentence += self.index_to_target[idx.item()]
                    print(f'Target prediction: {sentence}')

                    self.model.eval()
                    # send test sentence to tokenizer with the form of a batch
                    source_sentence = ("should we go to the mall?", )
                    target_sentence = ("", )

                    for word_id in range(self.max_sequence_length):
                        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask \
                        = self.mask.create_masks(source_sentence, target_sentence)
                        encoder_self_attention_mask = encoder_self_attention_mask.to(self.device)
                        decoder_self_attention_mask = decoder_self_attention_mask.to(self.device)
                        decoder_cross_attention_mask = decoder_cross_attention_mask.to(self.device)
                        pred = self.model(source_sentence, target_sentence, device = self.device,
                                          encoder_self_attention_mask=encoder_self_attention_mask,
                                          decoder_self_attention_mask=decoder_self_attention_mask,
                                          decoder_cross_attention_mask=decoder_cross_attention_mask,
                                          enc_start_token=False,
                                          enc_end_token=False,
                                          dec_start_token=False,
                                          dec_end_token=False)
                        next_token_prob_distribution = pred[0][word_id]
                        next_token_index = torch.argmax(next_token_prob_distribution).item()
                        next_token = self.index_to_target[next_token_index]
                        if next_token == self.END_TOKEN:
                            break
                        target_sentence = (target_sentence[0] + next_token,)

                    print(f'Evaluation (should we go to the mall?): {target_sentence}')
                    print("-" * 50)

        self.save_model()

if __name__ == "__main__":
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.train()