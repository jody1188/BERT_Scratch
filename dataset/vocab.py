import os

from tokenizers import ByteLevelBPETokenizer


def Train_Tokenizer(data_dir, tokenizer_dir, vocab_size, seq_len, min_frequency):

    tokenizer = ByteLevelBPETokenizer()

    #file_paths = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
    special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "<MASK>"]

    tokenizer.train(files = data_dir, vocab_size = vocab_size, min_frequency = min_frequency,
                                                                special_tokens = special_tokens)

    tokenizer.save_model(tokenizer_dir)
    tokenizer = ByteLevelBPETokenizer(tokenizer_dir + '/vocab.json', tokenizer_dir + '/merges.txt')

    tokenizer.enable_truncation(max_length = seq_len)
    tokenizer.save(tokenizer_dir + '/BPETokenizer.json')


