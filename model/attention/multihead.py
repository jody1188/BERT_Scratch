import torch
import torch.nn as nn

from .singlehead import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):


    def __init__(self, emb_dim, n_heads):
        super().__init__()

        assert emb_dim % n_heads == 0

        self.dim_k = emb_dim // n_heads
        self.n_heads = n_heads

        self.qkv_layers = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(3)])

        self.self_attention = ScaledDotProductAttention()

        self.output_layer = nn.Linear(emb_dim, emb_dim)


    def forward(self, query, key, value, mask = None):

        batch_size = query.size(0)

        query, key, value = [x(y).view(batch_size, -1, self.n_heads, self.dim_k).transpose(1, 2)
                             for x, y in zip(self.qkv_layers, (query, key, value))]

        attn, attn_prob = self.self_attention(query, key, value, mask = mask)

        attn_output = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.dim_k)
        attn_output = self.output_layer(attn_output)

        return attn_output 
