# File: features.py
# Creation: Saturday September 26th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# img2poem package
from img2poem.tokenizer import pad_sequences


class FeaturesDataset(Dataset):
    def __init__(self, filename, tokenizer=None, vocab_size=3_000, max_seq_len=128):
        self.filename = filename
        self.tokenizer = tokenizer
        self.data = pd.DataFrame(pd.read_pickle(filename))
        ids = []
        poems = []
        features = []
        for _, row in tqdm(self.data.iterrows(), desc='Loading', position=0, leave=True, total=len(self.data)):
            id = row.id
            poem = row.poem.replace("\n", " ; ")
            feature = row.feature
            ids.append(id)
            poems.append(poem)
            features.append(feature)
        self.ids = torch.tensor(ids)
        self.tokenizer.build_vocab(poems, vocab_size=vocab_size)
        tokens, tokens_ids, masks = pad_sequences(poems, self.tokenizer,
                                      max_seq_len=max_seq_len,
                                      sos_token="<sos>",
                                      eos_token="<eos>",
                                      pad_token="<pad>")
        self.tokens_ids = torch.tensor(tokens_ids)
        self.tokens = tokens
        self.masks = torch.tensor(masks)
        self.lengths = torch.tensor([np.sum(mask) for mask in masks])
        self.features = torch.tensor(features)

    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda value: value[3], reverse=True)
        ids = torch.tensor([x[0] for x in batch])
        tokens = [x[1] for x in batch]
        token_ids = torch.stack([x[2] for x in batch])
        lengths = torch.stack([x[3] for x in batch])
        features = torch.stack([x[4] for x in batch])
        return (ids, tokens, token_ids, lengths, features)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], self.tokens[index], self.tokens_ids[index], self.lengths[index], self.features[index]
