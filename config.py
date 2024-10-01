class Config():
    """
    This class contains Variable Required To Train a Transformer-based Translator Model.
    """

    # data config
    source_file = 'data/english.txt'
    target_file = 'data/kannada.txt'

    START_TOKEN = ''
    PADDING_TOKEN = ''
    END_TOKEN = ''

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

    source_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                          '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                          ':', '<', '=', '>', '?', '@',
                          '[', ']', '^', '_', '`',
                          'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                          'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                          'y', 'z',
                          '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN]

    index_to_target = {k: v for k, v in enumerate(target_vocabulary)}
    target_to_index = {v: k for k, v in enumerate(target_vocabulary)}
    index_to_source = {k: v for k, v in enumerate(source_vocabulary)}
    source_to_index = {v: k for k, v in enumerate(source_vocabulary)}

    TOTAL_SENTENCES = 200000
    PERCENTILE = 97

    max_sequence_size = 200

    batch_size = 16

    # model config
    d_model = 512
    ffn_hidden = 1024
    num_heads = 8
    vocab_size = len(target_vocabulary)
    num_layers = 1
    drop_prob = 0.1

    # mask config
    NEG_INFTY = 1e-9

    # train config
    num_epochs = 10
    lr = 1e-4

    enc_start_token = False
    enc_end_token = False
    dec_start_token = True
    dec_end_token = True