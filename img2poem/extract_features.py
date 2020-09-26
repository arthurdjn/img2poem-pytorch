# File: extract_features.py
# Creation: Saturday September 26th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
from tqdm import tqdm
from collections import defaultdict
import pickle
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# img2poem package
from img2poem.datasets import PoemUniMDataset, PoemMultiMDataset
from img2poem.nn import PoemEmbedder, ImageEmbedder


def extract_poem_features(filename, embedder_path, device="cuda"):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = PoemUniMDataset(filename, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    embedder = PoemEmbedder().to(device)
    embedder.load_state_dict(torch.load(embedder_path))
    features = {
        "id": defaultdict(list),
        "feature": defaultdict(list),
        "poem": defaultdict(list)
    }
    with torch.no_grad():
        for id, tokens_id, mask in tqdm(loader):
            tokens_id = tokens_id.to(device)
            mask = mask.to(device)
            feature = embedder(tokens_id, mask)
            feature = feature.cpu().squeeze(0).numpy()
            id = int(id.squeeze(0).numpy().item())
            features["id"].append(id)
            features["feature"].append(feature)
            features["poem"].append(dataset._df.iloc[id].poem)

    with open('data/poem_features.pkl', 'wb') as f:
        pickle.dump(features, f)


def extract_image_features(filename, image_dir, embedder_path, device="cuda"):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = PoemMultiMDataset(filename, image_dir, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    embedder = ImageEmbedder().to(device)
    embedder.load_state_dict(torch.load(embedder_path))
    features = {
        "id": defaultdict(list),
        "feature": defaultdict(list),
        "poem": defaultdict(list)
    }
    with torch.no_grad():
        for id, _, _, image in tqdm(loader):
            image = image.to(device)
            feature = embedder(image)
            feature = feature.cpu().squeeze(0).numpy()
            id = int(id.squeeze(0).numpy().item())
            features["id"].append(id)
            features["feature"].append(feature)
            features["poem"].append(dataset._df.iloc[id].poem)

    with open('data/image_features.pkl', 'wb') as f:
        pickle.dump(features, f)
