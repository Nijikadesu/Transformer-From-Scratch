from config import Config

class DataProcessor():
    def __init__(self, cfg):
        self.cfg = cfg

    def is_valid_tokens(self, sentence, vocab):
        for token in list(set(sentence)):
            if token not in vocab:
                return False
        return True

    def is_valid_length(self, sentence):
        return len(list(sentence)) < (self.cfg.max_sequence_length - 1) # leave 1 space for end token

    def read_data(self):
        with open(self.cfg.source_file, 'r', encoding='utf-8') as file:
            source_sentences = file.readlines()
        with open(self.cfg.target_file, 'r', encoding='utf-8') as file:
            target_sentences = file.readlines()

        source_sentences = source_sentences[: self.cfg.TOTAL_SENTENCES]
        target_sentences = target_sentences[: self.cfg.TOTAL_SENTENCES]
        source_sentences = [sentence.rstrip('\n').lower() for sentence in source_sentences]
        target_sentences = [sentence.rstrip('\n') for sentence in target_sentences]

        valid_sentence_indices = []

        for index in range(len(target_sentences)):
            target_sentence, source_sentence = target_sentences[index], source_sentences[index]
            if self.is_valid_length(source_sentence) and self.is_valid_length(target_sentence) \
            and self.is_valid_tokens(source_sentence, vocab=self.cfg.source_vocabulary) \
            and self.is_valid_tokens(target_sentence, vocab=self.cfg.target_vocabulary):
                valid_sentence_indices.append(index)

        print(f'Number of valid sentences: {len(valid_sentence_indices)}')

        source_sentences = [source_sentences[i] for i in valid_sentence_indices]
        target_sentences = [target_sentences[i] for i in valid_sentence_indices]

        return source_sentences, target_sentences