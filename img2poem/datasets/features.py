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
import torch
from torch.utils.data import Dataset

# img2poem package
from img2poem.tokenizer import pad_bert_sequences


class FeaturesDataset(Dataset):
    def __init__(self, filename, tokenizer=None, max_seq_len=128):
        self.filename = filename
        self.data = pd.read_pickle(filename)
        ids = []
        poems = []
        for _, row in tqdm(self.data.iterrows(), desc='Loading', position=0, leave=True, total=len(self.data)):
            id = row.id
            poem = row.poem.replace("\n", " ; ")
            poems.append(poem)
            ids.append(id)
        tokens_ids, masks = pad_bert_sequences(poems, tokenizer, max_seq_len=max_seq_len)
        self.ids = torch.tensor(ids)
        self.tokens_ids = torch.tensor(tokens_ids)
        self.masks = torch.tensor(masks)    
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], self.tokens_ids[index], self.masks[index]