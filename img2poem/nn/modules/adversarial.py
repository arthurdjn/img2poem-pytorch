# File: adversarial.py
# Creation: Saturday September 12th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# img2poem
from img2poem.nn.utils import normalize


class PoeticDecoder(nn.Module):
    """RNN-based Decoder for features from the poetic space.
    The decoder is based on a recurrent network, initialized by the features.

    """

    def __init__(self, vocab_size, hidden_dim, embedding_dim, num_layers=2, bidirectional=True,
                 features_dim=512, dropout=0.2, max_seq_len=128, token_sos_id=0, token_eos_id=1):
        super(PoeticDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        # The embedding layer should have `token_sos_id` and `token_eos_id` rows in the matrix
        self.token_sos_id = token_sos_id
        self.token_eos_id = token_eos_id
        self.embedding = nn.Embedding(features_dim, embedding_dim)
        self.rnn_cell = nn.GRUCell(features_dim, hidden_dim)
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if num_layers < 2 else dropout)
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, token_ids):
        """Forward pass used to train the embedding and recurrent layer with tensors from the Poetic space,
        w.r.t. real poems from MultiM and UniM.

        .. note::
            This method should only be used while training.
            For poem generation and tests, please use the ``generate()`` method.

        Args:
            features (torch.tensor): Tensor features from the Poetic space, of shape :math:`(B, F)`.
            token_ids (torch.tensor): Token ids from ground truth poems, of shape :math:`(B, T)`.

        Returns:
            Generated poem, of shape :math:`(B, T)`.
        """
        # features = B, features_dim
        features = normalize(features)
        # use features from the poetic space as hidden state
        h, c = self.rnn_cell(features)
        # train the embedding layer
        embeddings = self.embedding(token_ids)
        # embedding = B, hidden_dim, embedding_dim
        embeddings = self.dropout(embeddings)
        packed_output, (_, _) = self.rnn(embeddings, (h, c))
        sequences, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # sequences = B, max_seq_len
        out = self.dropout(sequences)
        out = self.fc(out)
        # out = B, vocab_size
        return out

    def generate(self, features, temperature=1):
        """
        Generate captions for given image features using greedy search.
        Code translated from `img2poem <https://github.com/researchmm/img2poem/blob/master/code/src/model.py>`__ repository.

        .. note::
            To use this method, make sure the embedding layer is trained.

        Args:
            features (torch.tensor): Features from the Poetic space, of shape :math:`(B, F)`.
            temperature (float, optional): How close the generated should be from ground truth poems. Default to 1.

        Returns:
            Generated poem, of shape :math:`(B, T)`.
        """
        features = normalize(features)
        batch_size = features.shape[0]

        sampled_ids = []
        # use [SOS] as init input
        start = torch.full((batch_size, 1), self.token_sos_id, dtype=torch.int).long().to(self.device)  # start symbol index is 1
        inputs = self.embed(start)  # inputs: (batch_size, 1, embed_size)

        # use features from the poetic space as hidden state
        (h, c) = self.rnn_cell(features)
        for i in range(self.max_seq_len):
            lstm_outputs, (h, c) = self.rnn(inputs, (h, c))
            outputs = self.linear(lstm_outputs.squeeze(1))
            weights = torch.functional.softmax(outputs / temperature, dim=1)
            predicted = torch.multinomial(weights, 1).squeeze(-1)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


class Discriminator(nn.Module):
    """Discriminator used to classify 
    generated poems from real ones.

    * :attr:`vocab_size` (int): Size of the vocabulary used.

    * :attr:`hidden_dim` (int): Dimension of the RNN hidden dimension.

    * :attr:`embedding_dim` (int): Dimension of the embedding layer.

    * :attr:`embedding` (torch.nn.Embedding): Embedding layer used to map words to a textual space.

    * :attr:`rnn` (torch.nn.Module): Recurrent Layer used to embed a poem.

    * :attr:`fc` (torch.nn.Linear): Linear layer used to classify real poems to generated poems.

    * :attr:`dropout` (float): Ratio of skipped connections between the ``rnn`` and ``fc`` layers.

    """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=3, features_dim=512, bidirectional=True, dropout=0.2):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, lengths):
        embeddings = self.embedding(sequence)
        packed_sequence = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # packed_sequence = B, max_seq_len, embedding_dim
        _, (hidden, _) = self.rnn(packed_sequence)
        # hidden = B, hidden_dim
        out = self.dropout(hidden)
        out = self.fc(out)
        # out = B, C
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
