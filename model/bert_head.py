import torch
import torch.nn as nn

from .bert import BERT



class BERTMLM(nn.Module):


    def __init__(self, emb_dim, vocab_size):

        super().__init__()

        self.linear_layer = nn.Linear(emb_dim, vocab_size)

        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, input):

        mlm_output = self.linear_layer(input)
        mlm_output = self.softmax(mlm_output)

        return mlm_output



class BERTNSP(nn.Module):


    def __init__(self, emb_dim):

        super().__init__()

        self.linear_layer = nn.Linear(emb_dim, 2)

        self.softmax = nn.LogSoftmax(dim = -1)

    def forward(self, input):

        nsp_output = self.linear_layer(input[:, 0])
        nsp_output = self.softmax(nsp_output)

        return nsp_output


class BERTLM(nn.Module):
    
    def __init__(self, bert : BERT, vocab_size, emb_dim):

        super().__init__()

        self.bert = bert
        self.bertmlm = BERTMLM(emb_dim, vocab_size)
        self.bertnsp = BERTNSP(emb_dim)


    def forward(self, input, segment_label):

        lm_output = self.bert.forward(input, segment_label)

        mlm_output = self.bertmlm(lm_output)
        nsp_output = self.bertnsp(lm_output)

        return mlm_output, nsp_output


class BERTClassifier(nn.Module):

    def __init__(self, bert : BERT, emb_dim, n_labels, dropout_prob):
        super().__init__()

        self.bert = bert
        self.linear_layer = nn.Linear(emb_dim, emb_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(emb_dim, n_labels)

    def forward(self, input, segment_label):

        bert_output = self.bert(input, segment_label)
        
        pooling = self.linear_layer(bert_output[:, 0])
        pooling = self.tanh(pooling)
        logits = self.classifier(self.dropout(pooling))

        return logits

