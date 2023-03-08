import torch
import torch.nn as nn

from .encoder import EncoderBlock
from .embedding import BERTEmbedding


class BERT(nn.Module):


    def __init__(self, vocab_size, seq_len, emb_dim, ff_dim, n_layers, n_heads, dropout_prob, device):
        

        super().__init__()

        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.n_heads = n_heads


        self.embedding = BERTEmbedding(vocab_size, emb_dim, seq_len, dropout_prob, device)

        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(emb_dim, n_heads, ff_dim, dropout_prob) for _ in range(n_layers)])


    def forward(self, input, segment_label):

        mask = (input > 0).unsqueeze(1).repeat(1, input.size(1), 1).unsqueeze(1)

        emb_output = self.embedding(input, segment_label)

        for encoder in self.encoder_blocks:
            bert_output = encoder.forward(emb_output, mask)

        return bert_output
