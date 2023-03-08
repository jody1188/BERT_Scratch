import math

import torch
import torch.nn as nn
import torch.nn.functional as F




class ScaledDotProductAttention(nn.Module):

    def __init__(self):

        super().__init__()


    def forward(self, query, key, value, mask = None):

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:

            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_prob = F.softmax(attn_scores, dim = -1)

        attn = torch.matmul(attn_prob, value)

        return attn, attn_prob
