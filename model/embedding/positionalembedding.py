import math

import torch
import torch.nn as nn



class PositionalEmbedding(nn.Module):

    def __init__(self, emb_dim, seq_len, device):
        super().__init__()
        
        self.encoding = torch.zeros(seq_len, emb_dim, device=device)
        self.encoding.requires_grad = False
        
        pos = torch.arange(0,seq_len,device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0,emb_dim,step=2,device=device).float()
        
        # self.encoding = (sequence_length, hidden_size)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i/emb_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i/emb_dim)))
        
    def forward(self, tensor):
        batch_size, sequence_length = tensor.size()
        
        # (sequence_length, hidden_size)
        return self.encoding[:sequence_length, :]