# File: utils.py
# Creation: Saturday September 12th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
from pytorch_pretrained_bert import BertTokenizer
import pandas as pd
from torch.utils.data import TensorDataset
import requests
import os
import torch


def download_image(url, outname, outdir='./'):
    """Download an image from a URL.
    Original code from a `post <https://stackoverflow.com/a/30229298>`__ on Stackoverflow.

    Args:
        url (str): URL to the image.
        outname (str): Name of the image, with its extension.
        outdir (str): Path to the saving directory.

    Example:
        >>> url = "http://farm4.staticflickr.com/3910/14393814286_c6dbcf7a92_z.jpg"
        >>> outname = "my_image.png"
        >>> download_image(url, outname)
    """
    with open(os.path.join(outdir, outname), 'wb') as handle:
        try:
            response = requests.get(url, stream=True)
            for block in response.iter_content(1024):
                handle.write(block)
        except Exception as error:
            print(f"WARNING: An error occured. {error}"
                  f"Could not download the image {outdir}/{outname} from the URL {url}.")


def pad_bert_sequence(sequence, tokenizer, max_seq_len=256, cls_token='[CLS]', sep_token='[SEP]'):
    """Pad a sequence (in text format) with the BERT tokenizer. It add a ``[CLS]`` and ``[SEP]`` tokens
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
    tokens = [cls_token] + tokens[0:max_seq_len-2] + [sep_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Pad the sequences
    tokens_ids_padded = tokens_ids + [0] * (max_seq_len - len(tokens_ids))
    attention_masks = [int(x > 0) for x in tokens_ids_padded]
    # To tensors
    tokens_ids = torch.tensor(tokens_ids_padded, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    return tokens_ids, attention_masks


def pad_bert_sequences(sequences, tokenizer, max_seq_len=6, cls_token='[CLS]', sep_token='[SEP]'):
    """Pad a list of sequences (in text format) with the BERT tokenizer. It add a ``[CLS]`` and ``[SEP]`` tokens
    at the begining and end of the truncated sentences (the sentences have ``max_seq_len`` length).

    Args:
        sequence (str): The sentences to tokenize.
        tokenizer (BertTokenizer): Bert tokenizer.
        max_seq_len (int, optional): Maximal length of a sequence. Defaults to 256.
        cls_token (str, optional): The Start of String token. Defaults to '[CLS]'.
        sep_token (str, optional): The Separator (or End of String) token. Defaults to '[SEP]'.

    Returns:
        tuple: tokenized sentences and attention masks.
    """
    # Tokenize the sequences
    tokens = [tokenizer.tokenize(sequence) for sequence in sequences]
    tokens = [[cls_token] + sequence[0:max_seq_len-2] + [sep_token] for sequence in tokens]
    tokens_ids = [tokenizer.convert_tokens_to_ids(tks) for tks in tokens]
    # Pad the sequences
    tokens_ids_padded = [tokens + [0] * (max_seq_len - len(tks)) for tks in tokens_ids]
    attention_masks_int = [[int(x > 0) for x in tks] for tks in tokens_ids_padded]
    # To tensors
    tokens_ids = torch.tensor(tokens_ids_padded, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks_int, dtype=torch.long)
    return tokens_ids, attention_masks
