import os
import json
import tqdm

from utils import arg_parse
from collections import namedtuple

import pandas as pd

from tokenizers import ByteLevelBPETokenizer

import torch
from torch.utils.data import DataLoader
from dataset import Train_Tokenizer, LMDataset, IMDBdataset


from model import BERT

from pretrain import Pretrainer
from finetune import Trainer


def main(train_mode):

    if train_mode == "pretrain":
        args = arg_parse()
        
        with open(args.cfgs, 'r') as f: 
            cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_dir = cfgs.data_dir
        tokenizer_dir = cfgs.tokenizer_dir
        vocab_size = cfgs.vocab_size
        seq_len = cfgs.seq_len
        min_freq = cfgs.min_freq

        print("Train Tokenizer")

        data_list = [os.path.join(data_dir,i) for i in os.listdir(data_dir)]

        Train_Tokenizer(data_list[0], tokenizer_dir, vocab_size, seq_len, min_freq)
        tokenizer = ByteLevelBPETokenizer(tokenizer_dir + '/vocab.json', tokenizer_dir + '/merges.txt')

        vocab = tokenizer.get_vocab()

        print("Preparing Pretrain Dataset and Split Dataset")

        pretrain_dataset = LMDataset(data_dir, tokenizer, seq_len, vocab)

        dataset_len = len(pretrain_dataset)
        valid_ratio = cfgs.valid_ratio
    
        train_dataset_len = int(dataset_len * (1 - valid_ratio))
        valid_dataset_len = dataset_len - train_dataset_len
        print(train_dataset_len)
        print(valid_dataset_len)
        print(dataset_len)
        print(train_dataset_len + valid_dataset_len)

        train_dataset, valid_dataset = torch.utils.data.random_split(pretrain_dataset, 
                                                            [train_dataset_len, valid_dataset_len])

        print("Finish!")


        print("Preparing Train Dataloader")

        batch_size = cfgs.batch_size
        num_workers = cfgs.num_workers


        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers)
        valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, num_workers = num_workers)

        print("Finish!")

        print("Build BERT Model")

        emb_dim = cfgs.emb_dim
        ff_dim = cfgs.ff_dim
        n_layers = cfgs.n_layers
        n_heads = cfgs.n_heads
        dropout_prob = cfgs.dropout_prob

        print(vocab_size)

        bert = BERT(vocab_size, seq_len, emb_dim, ff_dim, n_layers, n_heads, dropout_prob, device)

        print("Finsh!")

        checkpoint_dir = cfgs.checkpoint_dir
        learning_rate = cfgs.learning_rate
        adam_beta1 = cfgs.adam_beta1
        adam_beta2 = cfgs.adam_beta2
        betas = (adam_beta1, adam_beta2)
        weight_decay = cfgs.weight_decay
        warmup_steps = cfgs.warmup_steps
        epochs = cfgs.epochs
        save_epoch = cfgs.save_epoch
        steps = cfgs.steps
        log_freq = cfgs.log_freq


        pretrainer = Pretrainer(bert, emb_dim, vocab_size, epochs, save_epoch, checkpoint_dir, train_dataloader, valid_dataloader, 
                                                    learning_rate, betas, weight_decay, warmup_steps, log_freq, steps, device)

        print("Start Training!")

        pretrainer.training()

        print("End Training!")





    elif train_mode == "finetune":
        args = arg_parse()
        
        with open(args.cfgs, 'r') as f: 
            cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_dir = cfgs.data_dir
        tokenizer_dir = cfgs.tokenizer_dir
        vocab_size = cfgs.vocab_size
        seq_len = cfgs.seq_len
        min_freq = cfgs.min_freq

        print("Train Tokenizer")

        data_list = [os.path.join(data_dir,i) for i in os.listdir(data_dir)]

        df = pd.read_csv(data_list[1])

        #Train_Tokenizer(data_list[1], tokenizer_dir, vocab_size, seq_len, min_freq)
        tokenizer = ByteLevelBPETokenizer(tokenizer_dir + '/vocab.json', tokenizer_dir + '/merges.txt')
        vocab = tokenizer.get_vocab()
        
        print("Preparing Finetune Dataset and Split Dataset")

        finetune_dataset = IMDBdataset(df, tokenizer, seq_len, vocab)

        dataset_len = len(finetune_dataset)

        valid_ratio = cfgs.valid_ratio
    
        train_dataset_len = int(dataset_len * (1 - valid_ratio))
        valid_dataset_len = dataset_len - train_dataset_len

        train_dataset, valid_dataset = torch.utils.data.random_split(finetune_dataset, 
                                                            [train_dataset_len, valid_dataset_len])

        print("Finish!")

        print("Preparing Train Dataloader")

        batch_size = cfgs.batch_size
        num_workers = cfgs.num_workers

        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers)
        valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, num_workers = num_workers)

        print("Finish!")

        print("Build BERT Model")

        emb_dim = cfgs.emb_dim
        ff_dim = cfgs.ff_dim
        n_layers = cfgs.n_layers
        n_heads = cfgs.n_heads
        dropout_prob = cfgs.dropout_prob

        bert = BERT(vocab_size, seq_len, emb_dim, ff_dim, n_layers, n_heads, dropout_prob, device)
        checkpoint_dir = cfgs.checkpoint_dir
        learning_rate = cfgs.learning_rate
        adam_beta1 = cfgs.adam_beta1
        adam_beta2 = cfgs.adam_beta2
        betas = (adam_beta1, adam_beta2)
        weight_decay = cfgs.weight_decay
        warmup_steps = cfgs.warmup_steps
        epochs = cfgs.epochs
        save_epoch = cfgs.save_epoch
        steps = cfgs.steps
        log_freq = cfgs.log_freq


        finetuner = Trainer(bert, emb_dim, dropout_prob, epochs, save_epoch, checkpoint_dir, train_dataloader, valid_dataloader, 
                                                    learning_rate, betas, weight_decay, warmup_steps, log_freq, steps, device)

        print("Start Training!")

        finetuner.training()

        print("End Training!")

if __name__ == "__main__":

    train_mode = input("Enter Training Mode : ")

    main(train_mode)