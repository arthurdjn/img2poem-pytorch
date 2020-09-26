# File: poem.py
# Creation: Saturday September 12th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


r"""
This module defines the poem datasets used for textual embedding.
It mainly uses Pytorch but tensorFlow was used for padding sequences.

.. code-block:: python

    from img2poem.datasets import PoemMultiMDataset
    
    # Load the `multim_poem.json`
    # Or download it with `PoemMultiMDataset.download('.data')`
    PATH = 'data/poems/multim_poem.json'
    dataset = PoemMultiMDataset(PATH)
    
    # Split the dataset into training (80%), validation (10%) and testing (10%) parts
    dataset_train, dataset_valid = dataset.split(0.8)
    dataset_valid, dataset_test = dataset_valid.split(0.5)
    
"""


# Basic imports
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# img2poem package
from .utils import download_image, DEFAULT_TRANSFORM
from img2poem.tokenizer import pad_bert_sequences


class PoemUniMDataset(Dataset):
    r"""UniM Poem Dataset with masks used in the `paper <https://arxiv.org/abs/1804.08473>`__ 
    “Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training”
    from Liu, Bei et al. (2018).

    * :attr:`ids` (int): Identifier of the poem.

    * :attr:`tokens` (torch.tensor): Tokenized ids of a poem.

    * :attr:`masks` (torch.tensor): Tokenized ids masked.

    .. note::
        The default filename used to process the data is called ``unim_poem.json``.

    """

    url = 'https://github.com/researchmm/img2poem/blob/master/data/unim_poem.json'
    dirname = 'img2poem'
    name = 'unim'

    def __init__(self, filename, tokenizer=None, max_seq_len=128):
        super(PoemUniMDataset, self).__init__()
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained('bert-base-uncased')
        self.filename = filename
        self.data = pd.read_json(filename)
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


class PoemMultiMDataset(Dataset):
    r"""MultiM Poem Dataset with masks used in the `paper <https://arxiv.org/abs/1804.08473>`__ 
    “Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training”
    from Liu, Bei et al. (2018).

    * :attr:`id` (int): Identifier of the image & poem pair.

    * :attr:`tokens` (torch.tensor): Tokenized ids of a poem.

    * :attr:`masks` (torch.tensor): Tokenized ids masked.

    * :attr:`image` (torch.tensor): Matrix of the image in RGB format.

    .. note::
        The default filename used to process the data is called ``multim_poem.json``.
        The ``image_dir`` argument is used to locate the downloaded images.

    .. note::
        Download the images from the json file with the ``download`` class method.

    """

    url = 'https://raw.githubusercontent.com/researchmm/img2poem/master/data/multim_poem.json'
    dirname = 'img2poem'
    name = 'multim'

    def __init__(self, filename, image_dir, tokenizer=None, max_seq_len=128, transform=None):
        super(PoemMultiMDataset, self).__init__()
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained('bert-base-uncased')
        self.data = pd.read_json(filename)
        self.transform = transform or DEFAULT_TRANSFORM
        ids = []
        poems = []
        images = []
        for _, row in tqdm(self.data.iterrows(), desc='Loading', position=0, leave=True, total=len(self.data)):
            id = row.id
            poem = row.poem.replace("\n", " ; ")
            image_file = os.path.join(image_dir, f'{id}.jpg')
            try:
                image = self.transform(Image.open(image_file).convert('RGB'))
                ids.append(id)
                poems.append(poem)
                images.append(image)
            except Exception:
                pass

        self.ids = torch.tensor(ids)
        self.images = torch.stack(images)
        self.poems = poems
        token_ids, masks = pad_bert_sequences(poems, tokenizer, max_seq_len=max_seq_len)
        self.token_ids = torch.tensor(token_ids)
        self.masks = torch.tensor(masks)

    @classmethod
    def download(cls, root='.data', **kwargs):
        df = pd.read_json(cls.url)
        outdir = os.path.join(root, cls.dirname, cls.name)
        for _, row in tqdm(df.iterrows(), desc='Downloading', position=0, leave=True, total=len(df)):
            id = row.id
            url = row.image_url
            image_file = os.path.join(outdir, f'{id}.jpg')
            try:
                if not os.path.isfile(image_file):
                    download_image(url, image_file)
            except Exception:
                print(f"WARNING: Image {id} not downloaded from {url}.")

        return PoemMultiMDataset(cls.url, outdir, **kwargs)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], self.tokens_ids[index], self.masks[index], self.images[index]
