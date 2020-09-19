# File: adversarial.py
# Creation: Saturday September 12th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class PoeticDiscriminator(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes=2, feature_size=512, dropout=0.2):
        super(PoeticDiscriminator, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2 + feature_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        embeddings = self.embedding(x)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        _, (hidden, _) = self.rnn(packed)
        hidden = hidden.transpose(0, 1).contiguous().view(-1, self.hidden_size)
        out = self.dropout(hidden)
        out = self.fc(out)
        return out
    
    
class PoeticDecoder(nn.Module):
    def __init__(self):
        super(PoeticDecoder, self).__init__()
        pass

    def forward(self, x, lengths):
        pass