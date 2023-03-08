import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .module import PositionWiseFeedForward



class EncoderBlock(nn.Module):


    def __init__(self, emb_dim, n_heads, ff_dim, dropout_prob):


        super().__init__()

        self.multi_attention = MultiHeadAttention(emb_dim, n_heads)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_prob)

        self.feed_forward = PositionWiseFeedForward(emb_dim, ff_dim, dropout_prob)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, input, mask):

        res = input
        attn_output = self.multi_attention(input, input, input, mask = mask)
        attn_output = res + self.dropout1(self.gelu1(attn_output))

        res = attn_output
        ffn_output = self.feed_forward(attn_output)
        encoder_output = res + self.dropout2(self.gelu2(ffn_output))
        
        return encoder_output
