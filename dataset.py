from torch.utils.data import Dataset
from data_utils.read_file import DataProcessor
from data_utils.tokenizer import SentenceTokenizer

class TextDataset(Dataset):
    def __init__(self, cfg):
        dataprocessor = DataProcessor(cfg)
        source_sentence, target_sentence = dataprocessor.read_data()
        return source_sentence, target_sentence

    def __getitem__(self, idx):
        return self.source_sentence[idx], self.target_sentence[idx]

    def __len__(self):
        return len(self.source_sentence)