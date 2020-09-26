# File: tokenizer.py
# Creation: Saturday September 26th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
import nltk



def default_preprocess(sequence):
    return sequence.replace("\n", " ; ").replace("-", " - ")


def pad_bert_sequence(sequence, tokenizer, max_seq_len=512, cls_token='[CLS]', sep_token='[SEP]'):
    """Pad a single sequence (in text format) with the BERT tokenizer. It add a ``[CLS]`` and ``[SEP]`` tokens
    at the begining and end of the truncated sentence (the sentence has ``max_seq_len`` length).

    Args:
        sequence (str): The sentence to tokenize.
        tokenizer (BertTokenizer): Bert tokenizer.
        max_seq_len (int, optional): Maximal length of a sequence. Defaults to 256.
        cls_token (str, optional): The Start of String token. Defaults to '[CLS]'.
        sep_token (str, optional): The Separator (or End of String) token. Defaults to '[SEP]'.

    Returns:
        tuple: tokenized sentence and attention mask.
    """
    # Tokenize the sequence
    tokens = tokenizer.tokenize(sequence)
    # Truncate and add SOS and EOS special tokens
    sequence = [cls_token] + tokens[0:max_seq_len-2] + [sep_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Pad the sequences
    tokens_ids_padded = tokens_ids + [0] * (max_seq_len - len(tokens_ids))
    attention_masks = [int(x > 0) for x in tokens_ids_padded]
    return tokens_ids_padded, attention_masks


def pad_bert_sequences(sequences, tokenizer, max_seq_len=512, cls_token='[CLS]', sep_token='[SEP]'):
    """Pad a list of sequences (in text format) with the BERT tokenizer. It add a ``[CLS]`` and ``[SEP]`` tokens
    at the begining and end of the truncated sentences (the sentences have ``max_seq_len`` length).

    Args:
        sequences (list): The sentences to tokenize.
        tokenizer (BertTokenizer): Bert tokenizer.
        max_seq_len (int, optional): Maximal length of a sequence. Defaults to 256.
        cls_token (str, optional): The Start of String token. Defaults to '[CLS]'.
        sep_token (str, optional): The Separator (or End of String) token. Defaults to '[SEP]'.

    Returns:
        tuple: tokenized sentences and attention masks.
    """
    tokens_ids_padded = []
    attention_masks = []
    for sequence in sequences:
        tokens_ids, masks = pad_bert_sequence(sequence, tokenizer, 
                                              max_seq_len=max_seq_len, 
                                              cls_token=cls_token, 
                                              sep_token=sep_token)
        tokens_ids_padded.append(tokens_ids)
        attention_masks.append(masks)
    return tokens_ids_padded, attention_masks


class Tokenizer(object):
    def __init__(self, vocab=None, preprocess=None):
        self.vocab = vocab or OrderedDict()
        self.ids = OrderedDict()
        self.preprocess = preprocess or default_preprocess

    def build_vocab(self, sentences, vocab_size=None):
        freq = OrderedDict()
        for sentence in sentences:
            sentence = self.preprocess(sentence)
            for word in nltk.word_tokenize(sentence):
                freq[word] = freq.get(word, 0) + 1

        freq = sorted(freq.items(), key=lambda item: -item[1])
        for i, (key, _) in enumerate(freq[:vocab_size or -1]):
            self.vocab[key] = i
            self.ids[i] = key

    def tokenize(self, sentence):
        sentence = self.preprocess(sentence)
        return nltk.word_tokenize(sentence)

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.ids[i] for i in ids]
