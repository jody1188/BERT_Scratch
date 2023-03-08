import torch
from torch import nn


class PositionWiseFeedForward(nn.Module):

    def __init__(self, emb_dim, ff_dim, dropout_prob):

        super().__init__()

        self.gelu = nn.GELU()

        self.linear_layer1 = nn.Linear(emb_dim, ff_dim)
        self.linear_layer2 = nn.Linear(ff_dim, emb_dim)

        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, input):
        
        ff_output = self.linear_layer1(input)
        ff_output = self.gelu(ff_output)
        ff_output = self.linear_layer2(ff_output)
        ff_output = self.dropout(ff_output)

        return ff_output