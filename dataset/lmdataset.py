import os
import tqdm
import random

import numpy as np

import torch
from torch.utils.data import Dataset


class LMDataset(Dataset):

    def __init__(self, data_dir, tokenizer, seq_len, vocab):

        self.data_dir = data_dir
        self.seq_len = seq_len
        self.vocab = vocab
        self.tokenizer = tokenizer


        self.pad_index = self.vocab.get("<PAD>")
        self.mask_index = self.vocab.get("<MASK>")
        self.sos_index = self.vocab.get("<SOS>")
        self.eos_index = self.vocab.get("<EOS>")
        self.unk_index = self.vocab.get("<UNK>")

        #self.data_files = []

        #for filename in os.listdir(self.data_dir):
        #    f = os.path.join(data_dir, file_name)
        #    if os.path.isfile(f) and filename.endswith(".txt"):
        #        self.data_files.append(f)

        #for data in self.data_files:
        #    with open(os.path.join('dataset',path),'r') as f:

        data_paths = [os.path.join(self.data_dir,i) for i in os.listdir(self.data_dir)]


        with open(data_paths[1], "r") as f:
            self.corpus = f.read().split('\n')
            

    def __len__(self):

        return len(self.corpus)

        
    def __getitem__(self, item):
        
        prob = random.random()

        s1 = self.corpus[item].strip()
        s2_idx = item + 1

        nsp_label = 1

        if prob > 0.5:
            nsp_label = 0
            while (s2_idx == item + 1) or (s2_idx == item):
                s2_idx = random.randint(0, len(self.corpus))

        if s2_idx >= len(self.corpus):
            s2_idx = random.randint(0, len(self.corpus))
        
        s2 = self.corpus[s2_idx].strip()

        s1_tokens, s1_label = self.random_masking(s1)
        s2_tokens, s2_label = self.random_masking(s2)

        s1 = [self.sos_index] + s1_tokens + [self.eos_index]
        s2 = s2_tokens + [self.eos_index]

        s1_label = [self.pad_index] + s1_label + [self.pad_index]
        s2_label = s2_label + [self.pad_index]

        segment_label = ([0 for _ in range(len(s1))] + [1 for _ in range(len(s2))])[:self.seq_len]

        bert_input = (s1 + s2)[:self.seq_len]
        bert_label = (s1_label + s2_label)[:self.seq_len]

        padding = [self.pad_index for _ in range(self.seq_len - len(bert_input))]

        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "nsp_label": nsp_label}

        return {key: torch.tensor(value) for key, value in output.items()}




    def random_masking(self, sentence):

        tokens = self.tokenizer.encode(sentence).ids[1:-1]
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.8:
                    tokens[i] = self.mask_index

                elif prob < 0.9:
                    tokens[i] = random.randrange(self.tokenizer.get_vocab_size())

                else:
                    tokens[i] = self.vocab.get(token, self.unk_index)
                    
                output_label.append(self.vocab.get(token, self.unk_index))

            else:
                tokens[i] =  self.vocab.get(token, self.unk_index)
                output_label.append(0)

        return tokens, output_label


