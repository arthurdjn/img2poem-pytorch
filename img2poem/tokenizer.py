# File: tokenizer.py
# Creation: Saturday September 26th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
from collections import OrderedDict
import nltk


def default_preprocess(sequence):
    return sequence.replace("\n", " ; ").replace("-", " - ")


def pad_sequence(sequence, tokenizer, max_seq_len=512, sos_token='[CLS]', eos_token='[SEP]', pad_token='[PAD]'):
    """Pad a single sequence (in text format) with the BERT tokenizer. It add a ``[CLS]`` and ``[SEP]`` tokens
    at the begining and end of the truncated sentence (the sentence has ``max_seq_len`` length).

    Args:
        sequence (str): The sentence to tokenize.
        tokenizer (BertTokenizer): Bert tokenizer.
        max_seq_len (int, optional): Maximal length of a sequence. Defaults to 256.
        sos_token (str, optional): The Start of String token. Defaults to '[CLS]'.
        eos_token (str, optional): The Separator (or End of String) token. Defaults to '[SEP]'.

    Returns:
        tuple: tokenized sentence and attention mask.
    """
    # Tokenize the sequence
    tokens = tokenizer.tokenize(sequence)
    # Truncate and add SOS and EOS special tokens
    tokens = [sos_token] + tokens[0:max_seq_len - 2] + [eos_token]
    tokens_padded = tokens + [pad_token] * (max_seq_len - len(tokens))
    tokens_ids_padded = tokenizer.convert_tokens_to_ids(tokens_padded)
    # Pad the sequences
    pad_token_id = tokenizer.vocab[pad_token]
    attention_masks = [int(x != pad_token_id) for x in tokens_ids_padded]
    return tokens_padded, tokens_ids_padded, attention_masks


def pad_sequences(sequences, tokenizer, sos_token='[CLS]', eos_token='[SEP]', pad_token='[PAD]', max_seq_len=512):
    """Pad a list of sequences (in text format) with the BERT tokenizer. It add a ``[CLS]`` and ``[SEP]`` tokens
    at the begining and end of the truncated sentences (the sentences have ``max_seq_len`` length).

    Args:
        sequences (list): The sentences to tokenize.
        tokenizer (BertTokenizer): Bert tokenizer.
        max_seq_len (int, optional): Maximal length of a sequence. Defaults to 256.
        sos_token (str, optional): The Start of String token. Defaults to '[CLS]'.
        eos_token (str, optional): The Separator (or End of String) token. Defaults to '[SEP]'.

    Returns:
        tuple: tokenized sentences and attention masks.
    """
    tokens_padded = []
    tokens_ids_padded = []
    attention_masks = []
    for sequence in sequences:
        tokens, tokens_ids, masks = pad_sequence(sequence, tokenizer,
                                                 max_seq_len=max_seq_len,
                                                 sos_token=sos_token,
                                                 eos_token=eos_token,
                                                 pad_token=pad_token)
        tokens_padded.append(tokens)
        tokens_ids_padded.append(tokens_ids)
        attention_masks.append(masks)
    return tokens_padded, tokens_ids_padded, attention_masks


class Tokenizer(object):
    def __init__(self, vocab=None, preprocess=None,
                 sos_token='<sos>', eos_token='<eos>', unk_token='<unk>', pad_token='<pad>'):
        self.vocab = vocab or OrderedDict()
        self.ids = OrderedDict()
        self.preprocess = preprocess or default_preprocess
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token

    def build_vocab(self, sentences, vocab_size=None):
        # Add special tokens
        self.vocab[self.sos_token] = 0
        self.vocab[self.eos_token] = 1
        self.vocab[self.unk_token] = 2
        self.vocab[self.pad_token] = 3
        self.ids[0] = self.sos_token
        self.ids[1] = self.eos_token
        self.ids[2] = self.unk_token
        self.ids[3] = self.pad_token

        # Add words based on their frequencies
        freq = OrderedDict()
        for sentence in sentences:
            sentence = self.preprocess(sentence)
            for word in nltk.word_tokenize(sentence):
                freq[word] = freq.get(word, 0) + 1

        freq = sorted(freq.items(), key=lambda item: -item[1])
        for i, (key, _) in enumerate(freq[:vocab_size or -1]):
            self.vocab[key] = i + 4  # Because <sos>, <eos>, <unk>, <pad> tokens
            self.ids[i + 4] = key  # Because <sos>, <eos>, <unk>, <pad> tokens

    def tokenize(self, sentence):
        sentence = self.preprocess(sentence)
        tokens_ = nltk.word_tokenize(sentence)
        tokens = []
        for token in tokens_:
            if token in self.vocab:
                tokens.append(token)
            else:
                tokens.append(self.unk_token)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.ids[i] for i in ids]
