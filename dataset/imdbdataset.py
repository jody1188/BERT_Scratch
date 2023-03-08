import os
import csv
import tqdm
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset



class IMDBdataset(Dataset):

    def __init__(self, dataframe, tokenizer, seq_len, vocab):

        self.seq_len = seq_len
        self.vocab = vocab
        self.tokenizer = tokenizer

        self.data = dataframe

        self.review = str(self.data['review'].dropna())
        self.target = self.data['sentiment'].dropna()

        self.tokenizer = tokenizer


        self.pad_index = self.vocab.get("<PAD>")
        self.mask_index = self.vocab.get("<MASK>")
        self.sos_index = self.vocab.get("<SOS>")
        self.eos_index = self.vocab.get("<EOS>")
        self.unk_index = self.vocab.get("<UNK>")

            
    def __len__(self):

        return len(self.review)

        
    def __getitem__(self, item):

        s1, label = self.tokenizer.encode(self.review[item]).ids[1:-1], self.target[item]


        if label == "positive":
            label = 1
        else:
            label = 0
            
        segment_label = [0 for _ in range(len(s1))][:self.seq_len]

        input = s1[:self.seq_len]

        padding = [self.pad_index for _ in range(self.seq_len - len(input))]

        input.extend(padding), segment_label.extend(padding)

        output = {"input": input,
                  "label": label,
                  "segment_label": segment_label}

        return {key: torch.tensor(value) for key, value in output.items()}