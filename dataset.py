from torch.utils.data import Dataset
from data_utils.read_file import DataProcessor
from data_utils.tokenizer import SentenceTokenizer

class TextDataset(Dataset):
    """
    Practice: Dataset of our English-to-Kannda translation task.
    Data came from https://github.com/ajhalthor/Transformer-Neural-Network/tree/main, thanks!
    """
    def __init__(self, cfg):
        dataprocessor = DataProcessor(cfg)
        self.source_sentence, self.target_sentence = dataprocessor.read_data()

    def __getitem__(self, idx):
        return self.source_sentence[idx], self.target_sentence[idx]

    def __len__(self):
        return len(self.source_sentence)