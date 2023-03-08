import torch.nn as nn

from .positionalembedding import PositionalEmbedding

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, emb_dim):
        super(TokenEmbedding, self).__init__(vocab_size, emb_dim, padding_idx = 0)

class BERTEmbedding(nn.Module):


    def __init__(self, vocab_size, emb_dim, seq_len, dropout_prob, device):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, emb_dim)
        self.pos_emb = PositionalEmbedding(emb_dim, seq_len, device)
        self.seg_emb = nn.Embedding(3, emb_dim)
        self.tok_drop_out = nn.Dropout(dropout_prob)
        self.seg_drop_out = nn.Dropout(dropout_prob)
    
    def forward(self, input, segment_label):
        
        tok_emb = self.tok_emb(input)
        pos_emb = self.pos_emb(input)

        seg_emb = self.seg_emb(segment_label)
        
        return self.tok_drop_out(tok_emb) + self.seg_drop_out(seg_emb) + pos_emb
