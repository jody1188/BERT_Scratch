import os
import sys

import time
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from model import BERT, BERTClassifier
from utils import ScheduledOptim, load_checkpoint

import tqdm

import matplotlib.pyplot as plt


class Trainer:
    

    def __init__(self, bert : BERT, emb_dim, dropout_prob, epochs, save_epoch, checkpoint_dir, train_dataloader, valid_dataloader, 
                                                    learning_rate, betas, weight_decay, warmup_steps, log_freq, steps, device):

        

        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.bert = bert

        self.bertcls = BERTClassifier(self.bert, emb_dim, 2, dropout_prob).to(self.device)
        
        


        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optimizer = Adam(self.bertcls.parameters(), lr=learning_rate, betas = betas, weight_decay = weight_decay)
        self.optimizer_schedule = ScheduledOptim(self.optimizer, emb_dim, n_warmup_steps = warmup_steps)

        self.criterion = nn.CrossEntropyLoss()

        self.train_history = []
        self.valid_history = []

        self.log_freq = log_freq

        self.epochs = epochs
        self.save_epoch = save_epoch
        self.steps = steps

        print("Total Parameters:", sum([p.nelement() for p in self.bertcls.parameters()]))


    def save_checkpoint(self, epoch, train_loss, valid_loss):

            now = time.time()
        
            file_name = self.checkpoint_dir + f"Epoch{epoch}--Time{datetime.utcnow().timestamp():.0f}.pt"

            torch.save({
                'Epoch' : epoch, "Model_state_dict" : self.bertcls.state_dict(), 
            "Optimizer_state_dict" : self.optimizer.state_dict(),
            "Train_Loss" : train_loss, "Valid_loss" : valid_loss}, file_name)

            print(f"{file_name}Save!-{time.time() - now:.2f}")
            print('-' * 20)

    def calc_accuracy(X,Y):

        max_vals, max_indices = torch.max(X, 1)
        train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
        
        return train_acc
    


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

                train_acc = 0.0
                valid_acc = 0.0

                train_loss_per_epoch = 0.0
                valid_loss_per_epoch = 0.0

                self.bertcls.train()
        

                for batch_idx, batch in data_iterator:
                                     
                    batch = {key: value.to(self.device) for key, value in batch.items()}

                    logits = self.bertcls.forward(batch["input"], batch["segment_label"])
                    print(batch['input'].size())
                    print(batch["segment_label"].size())
                    print(logits.argmax(dim = 1))
                    self.optimizer.zero_grad()

                    loss = self.criterion(logits, batch["label"])

                    train_loss_per_iter = loss.item()
                    train_loss_per_epoch += train_loss_per_iter
                    

                    loss.backward()

                    self.optimizer_schedule.step_and_update_lr()

                    #train_acc += self.calc_accuracy(logits, batch["label"])

                    if ((batch_idx + 1) % self.steps == 0) and (batch_idx > 0):                        
                        sys.stdout.write(f"###Training### |  Epoch: {epoch + 1} |  Step: {(batch_idx + 1 / len(self.train_dataloader))} | Loss: {train_loss_per_iter}")
                        sys.stdout.write('\n')

                train_loss_per_epoch_mean = train_loss_per_epoch / len(self.train_dataloader)
                train_history.append(train_loss_per_epoch_mean)


                self.bertcls.eval()

                data_iterator = tqdm.tqdm(enumerate(self.valid_dataloader),
                              total=len(self.valid_dataloader),
                              bar_format="{l_bar}{r_bar}")


                for batch_idx, batch in data_iterator:

                    batch = {key: value.to(self.device) for key, value in batch.items()}

                    logits = self.bertcls.forward(batch["input"], batch["segment_label"])

                    loss = self.criterion(logits, batch["label"])

                    valid_loss_per_iter = loss.item()
                    valid_loss_per_epoch += valid_loss_per_iter

                    #train_acc += self.calc_accuracy(logits, batch["label"])

                valid_loss_per_epoch_mean = valid_loss_per_epoch / len(self.valid_dataloader)
                valid_history.append(valid_loss_per_epoch_mean)

                sys.stdout.write(f"Validation |  Epoch: {epoch + 1}  | Loss : {valid_loss_per_iter}")
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