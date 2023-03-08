import os
import sys

import time
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import BERTLM, BERT
from utils import ScheduledOptim

import tqdm

import matplotlib.pyplot as plt


class Pretrainer:
    

    def __init__(self, bert : BERT, emb_dim, vocab_size, epochs, save_epoch, checkpoint_dir, train_dataloader, valid_dataloader, 
                                                    learning_rate, betas, weight_decay, warmup_steps, log_freq, steps, device):

        

        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.bert = bert

        self.bertlm = BERTLM(self.bert, vocab_size, emb_dim).to(self.device)


        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optimizer = Adam(self.bertlm.parameters(), lr=learning_rate, betas = betas, weight_decay = weight_decay)
        self.optimizer_schedule = ScheduledOptim(self.optimizer, emb_dim, n_warmup_steps = warmup_steps)

        self.criterion = nn.NLLLoss()

        self.train_history = []
        self.valid_history = []

        self.log_freq = log_freq

        self.epochs = epochs
        self.save_epoch = save_epoch
        self.steps = steps

        print("Total Parameters:", sum([p.nelement() for p in self.bertlm.parameters()]))


    def save_checkpoint(self, epoch, train_loss, valid_loss):

            now = time.time()
        
            file_name = self.checkpoint_dir + f"Epoch{epoch}--Time{datetime.utcnow().timestamp():.0f}.pt"

            torch.save({
                'Epoch' : epoch, "Model_state_dict" : self.bertlm.state_dict(), 
            "Optimizer_state_dict" : self.optimizer.state_dict(),
            "Train_Loss" : train_loss, "Valid_loss" : valid_loss}, file_name)

            print(f"{file_name}Save!-{time.time() - now:.2f}")
            print('-' * 20)



    def training(self):

            print('-' * 20)

            print("Start Training!")

            save_epoch = 0

            train_history=[]
            valid_history=[]


            for epoch in range(self.epochs):
                

                sys.stdout.write(f"######### Epoch : {epoch + 1} / {self.epochs} ###########")

                data_iterator = tqdm.tqdm(enumerate(self.train_dataloader),
                              total=len(self.train_dataloader),
                              bar_format="{l_bar}{r_bar}")

                train_loss_per_epoch = 0.0

                self.bertlm.train()
        

                for batch_idx, batch in data_iterator:
                                     
                    batch = {key: value.to(self.device) for key, value in batch.items()}

                    mlm_output, nsp_output = self.bertlm.forward(batch["bert_input"], batch["segment_label"])
                    
                    self.optimizer.zero_grad()

                    nsp_loss = self.criterion(nsp_output, batch["nsp_label"])
             
                    mlm_loss = self.criterion(mlm_output.contiguous().view(-1, mlm_output.shape[-1]), 
                                                                           batch["bert_label"].contiguous().view(-1))


                    total_loss = nsp_loss + mlm_loss

                    total_loss.backward()

                    target = nsp_output.argmax(dim=-1).eq(batch["nsp_label"]).sum().item()


                    self.optimizer_schedule.step_and_update_lr()

                    train_mlm_loss_per_iter = mlm_loss.item()
                    train_nsp_loss_per_iter = nsp_loss.item()
                    train_loss_per_iter = total_loss.item()
                    train_loss_per_epoch += train_loss_per_iter


                    if ((batch_idx + 1) % self.steps == 0) and (batch_idx > 0):                        
                        sys.stdout.write(f"###Training### |  Epoch: {epoch + 1} |  Step: {(batch_idx + 1 / len(self.train_dataloader))} | MLM loss: {train_mlm_loss_per_iter} | NSP loss: {train_nsp_loss_per_iter} | Total loss : {train_loss_per_iter}")
                        sys.stdout.write('\n')

                self.optimizer_schedule.step_and_update_lr()

                train_loss_per_epoch_mean = train_loss_per_epoch / len(self.train_dataloader)
                train_history.append(train_loss_per_epoch_mean)


                self.bertlm.eval()

                data_iterator = tqdm.tqdm(enumerate(self.valid_dataloader),
                              total=len(self.valid_dataloader),
                              bar_format="{l_bar}{r_bar}")

                valid_loss_per_epoch = 0.0

                for batch_idx, batch in data_iterator:

                    batch = {key: value.to(self.device) for key, value in batch.items()}
                    
                    mlm_output, nsp_output = self.bertlm.forward(batch["bert_input"], batch["segment_label"])

                    nsp_loss = self.criterion(nsp_output, batch["nsp_label"])

                    mlm_loss = self.criterion(mlm_output.contiguous().view(-1, mlm_output.shape[-1]), 
                                                                           batch["bert_label"].contiguous().view(-1))
                    mlm_loss = self.criterion(mlm_output.contiguous().view(-1, mlm_output.shape[-1]), 
                                                                           batch["bert_label"].contiguous().view(-1))
                    total_loss = nsp_loss + mlm_loss

                    valid_mlm_loss_per_iter = mlm_loss.item()
                    valid_nsp_loss_per_iter = nsp_loss.item()
                    valid_loss_per_iter = total_loss.item()
                    valid_loss_per_epoch += valid_loss_per_iter

                valid_loss_per_epoch_mean = valid_loss_per_epoch / len(self.valid_dataloader)
                valid_history.append(valid_loss_per_epoch_mean)

                sys.stdout.write(f"Validation |  Epoch: {epoch + 1} | MLM loss : {valid_mlm_loss_per_iter} | NSP loss : {valid_nsp_loss_per_iter} | Total loss : {valid_loss_per_iter}")
                sys.stdout.write('\n')      

                if epoch % self.save_epoch == 0:
                    self.save_checkpoint(epoch, train_loss_per_epoch, valid_loss_per_epoch)

            print("Complete Pretraing BERT!")
            print(f"Total Epoch : {self.epochs}--Training Average Loss : {train_loss_per_epoch_mean}--Validation Average Loss : {valid_loss_per_epoch_mean}")
            
            self.loss_plot(train_history, valid_history, self.checkpoint_dir)



    def loss_plot(self, train_history, valid_history, save_path):
        
        epoch_size = np.linspace(0, self.epochs, self.epochs)

        plt.plot(epoch_size, np.array(train_history), label = "Training History")
        plt.plot(epoch_size, np.array(train_history), label = "Training History")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plot_dir = save_path 

        plt.savefig(plot_dir + "loss_plot.png")
        print("Save Plot!")
                    

                    