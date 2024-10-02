import torch

class Config():
    """
    This class contains Variable Required To Train a Transformer-based Translator Model.
    """

    """ data config """
    source_file = 'data/english.txt'  # language you want to translate from | source language
    target_file = 'data/kannada.txt'  # language you want to translate to | target language

    START_TOKEN = "S"  # start token
    PADDING_TOKEN = "P"  # padding token
    END_TOKEN = "E"  # end token

    # target language dictionary
    target_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                          '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '<', '=', '>', '?', 'ˌ',
                          'ँ', 'ఆ', 'ఇ', 'ా', 'ి', 'ీ', 'ు', 'ూ',
                          'ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ೠ', 'ಌ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ',
                          'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ',
                          'ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ',
                          'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ',
                          'ತ', 'ಥ', 'ದ', 'ಧ', 'ನ',
                          'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ',
                          'ಯ', 'ರ', 'ಱ', 'ಲ', 'ಳ', 'ವ', 'ಶ', 'ಷ', 'ಸ', 'ಹ',
                          '಼', 'ಽ', 'ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೃ', 'ೄ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ', '್', 'ೕ', 'ೖ', 'ೞ',
                          'ೣ', 'ಂ', 'ಃ',
                          '೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯', PADDING_TOKEN, END_TOKEN]

    # source language dictionary
    source_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                          '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                          ':', '<', '=', '>', '?', '@',
                          '[', ']', '^', '_', '`',
                          'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                          'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                          'y', 'z',
                          '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

    # translate token-id into word / word into token-id
    index_to_target = {k: v for k, v in enumerate(target_vocabulary)}
    target_to_index = {v: k for k, v in enumerate(target_vocabulary)}
    index_to_source = {k: v for k, v in enumerate(source_vocabulary)}
    source_to_index = {v: k for k, v in enumerate(source_vocabulary)}

    TOTAL_SENTENCES = 200000  # limit the amount of sentence you read from data file
    PERCENTILE = 97  # not used

    max_sequence_length = 200  # limit the max sequence length (word num per sentence) your transformer model can read

    batch_size = 16  # batch size of your data in data-loader

    """ model config """
    d_model = 512  # dimension of word embeddings
    ffn_hidden = 1024  # num of neaurons in the ffn hidden layer
    num_heads = 8  # num of heads in the attention calculation
    vocab_size = len(target_vocabulary)  # size of your target language dictionary
    num_layers = 1  # num of transformer encoder-decoder architecture
    drop_prob = 0.1   # prob of drop-out layers

    """ mask config """
    NEG_INFTY = 1e-9  # we don't usually set mask value to -inf, instead we use a very small negative number

    """ train config """
    num_epochs = 10  # num of train epochs
    lr = 1e-4  # learning rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # device to train on

    model_path = './model.pth'  # where to save model weights