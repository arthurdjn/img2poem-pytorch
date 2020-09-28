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

# img2poem
from img2poem.nn.utils import normalize


class FeaturesDecoder(nn.Module):
    """RNN-based Decoder for features from the poetic space.
    The decoder is based on a recurrent network, initialized by the features.

    """

    def __init__(self, vocab_size, hidden_dim, embedding_dim, num_layers=1, bidirectional=True,
                 features_dim=512, dropout=0.2, max_seq_len=128, sos_token_id=0, eos_token_id=1):
        super(FeaturesDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        # The embedding layer should have `sos_token_id` and `eos_token_id` rows in the matrix
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_cell = nn.LSTMCell(features_dim, hidden_dim)
        self.rnn = nn.LSTM(embedding_dim,
                          hidden_dim,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if num_layers < 2 else dropout)
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, token_ids, lengths):
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
        # train the embedding layer
        embedded = self.embedding(token_ids)
        # embedded = B, hidden_dim, embedding_dim
        embedded = self.dropout(embedded)
        # use features from the poetic space as hidden state
        h, c = self.rnn_cell(features)
        # h = B, hidden_dim | c = B, hidden_dim
        h = h.unsqueeze(0)  # h = 1, B, hidden_dim
        c = c.unsqueeze(0)  # c = 1, B, hidden_dim
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        packed_output, (_, _) = self.rnn(packed_embeddings, (h, c))
        sequences, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=token_ids.size(1)-1)
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
        # use <sos> as init input
        start = torch.full((batch_size, 1), self.sos_token_id, 
                           dtype=torch.int, device=features.device).long()
        
        # Predict the next sentences
        with torch.no_grad():
            inputs = self.embedding(start)
            # inputs = B, 1, embedding_dim
            # use features from the poetic space as hidden state
            (h, c) = self.rnn_cell(features)
            # h = B, hidden_dim | c = B, hidden_dim
            h = h.unsqueeze(0)  # h = 1, B, hidden_dim
            c = c.unsqueeze(0)  # c = 1, B, hidden_dim
            for i in range(self.max_seq_len):
                lstm_outputs, (h, c) = self.rnn(inputs, (h, c))
                outputs = self.fc(lstm_outputs.squeeze(1))
                weights = torch.nn.functional.softmax(outputs / temperature, dim=1)
                predicted = torch.multinomial(weights, 1).squeeze(-1)
                sampled_ids.append(predicted)
                inputs = self.embedding(predicted)
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

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=2, bidirectional=True, dropout=0.2):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2 if bidirectional else hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, lengths):
        embedded = self.embedding(sequence[:, 1:])  # Skip <sos> tokens
        embedded = self.dropout(embedded)
        packed_sequence = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        # packed_sequence = B, max_seq_len, embedding_dim
        _, (hidden, _) = self.rnn(packed_sequence)
        # hidden = bidirectional, B, hidden_dim
        hidden = hidden.transpose(0, 1).contiguous().view(-1, hidden.size(2) * hidden.size(0))
        # hidden = B, bidirectional * hidden_dim
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
