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


class PoeticDecoder(nn.Module):
    def __init__(self):
        super(PoeticDecoder, self).__init__()
        pass

    def forward(self, x, lengths):
        pass


class PoeticDiscriminator(nn.Module):
    """Poetic Discriminator used to classify generated poem from the poetic space to real poems.

    * :attr:`vocab_size` (int): Size of the vocabulary used.

    * :attr:`hidden_dim` (int): Dimension of the RNN hidden dimension.

    * :attr:`embedding_dim` (int): Dimension of the embedding layer.

    * :attr:`embedding` (torch.nn.Embedding): Embedding layer used to map words to a textual space.

    * :attr:`rnn` (torch.nn.Module): Recurrent Layer used to embed a poem.

    * :attr:`fc` (torch.nn.Linear): Linear layer used to classify real poems to generated poems.

    * :attr:`dropout` (float): Ratio of skipped connections between the ``rnn`` and ``fc`` layers.

    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=2, poetic_dim=512, dropout=0.2):
        super(PoeticDiscriminator, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_dim
        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2 + poetic_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        embeddings = self.embedding(x)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        _, (hidden, _) = self.rnn(packed)
        hidden = hidden.transpose(0, 1).contiguous().view(-1, self.hidden_dim)
        out = self.dropout(hidden)
        out = self.fc(out)
        return out


class GeneratorFromImage(nn.Module):
    def __init__(self):
        super(GeneratorFromImage, self).__init__()
        pass

    def forward(self, images):
        pass


class GeneratorFromPoem(nn.Module):
    def __init__(self):
        super(GeneratorFromPoem, self).__init__()
        pass

    def forward(self, poems):
        pass
