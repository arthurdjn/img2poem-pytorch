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
from pytorch_pretrained_bert.file_utils import filename_to_url
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torch
import torchtext.data
from torch.utils.data import Dataset


# img2poem package
from .utils import download_image, pad_bert_sequence, pad_bert_sequences


class PoemUniMDataset(torchtext.data.Dataset):
    r"""UniM Poem Dataset used in the `paper <https://arxiv.org/abs/1804.08473>`__ 
    “Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training”
    from Liu, Bei et al. (2018).

    * :attr:`id` (int): Identifier of the poem.

    * :attr:`poem` (list(str)): Tokenized poem.

    .. note::
        The default filename used to process the data is called ``unim_poem.json``.

    """

    url = ['https://github.com/researchmm/img2poem/blob/master/data/unim_poem.json']
    dirname = 'img2poem'
    name = 'unim'

    def __init__(self, filename, tokenizer=None):
        # Define fields that compose an example
        field_id = torchtext.data.Field(batch_first=True, lower=False, include_lengths=False, pad_token=None)
        field_poem = torchtext.data.Field(batch_first=True, lower=False, include_lengths=True, pad_token=None, tokenize=tokenizer)
        fields = [("id", field_id), ("poem", field_poem)]

        df = pd.read_json(filename)
        examples = []
        for _, row in tqdm(df.iterrows(), position=0, leave=True, total=len(df)):
            examples.append(torchtext.data.Example.fromlist([row.id, row.poem], fields))
        super(PoemUniMDataset, self).__init__(examples, fields)


class PoemMultiMDataset(torchtext.data.Dataset):
    r"""MultiM Poem Dataset used in the `paper <https://arxiv.org/abs/1804.08473>`__ 
    “Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training”
    from Liu, Bei et al. (2018).

    * :attr:`id` (int): Identifier of the image & poem pair.

    * :attr:`url` (str): URL to the image.

    * :attr:`image` (np.ndarray): Matrix of the image in RGB format.

    * :attr:`poem` (list(str)): Tokenized poem.

    .. note::
        The default filename used to process the data is called ``multim_poem.json``.
        The ``image_dir`` argument is used the location of the downloaded images.
        
    .. note::
        Download the images from the csv file with the ``download`` method.

    """

    url = 'https://raw.githubusercontent.com/researchmm/img2poem/master/data/multim_poem.json'
    dirname = 'img2poem'
    name = 'multim'

    def __init__(self, filename, image_dir, tokenizer=None, transform=None):
        # Define fields that compose an example
        field_id = torchtext.data.Field(batch_first=True, lower=False, include_lengths=False, pad_token=None)
        field_url = torchtext.data.Field(batch_first=True, lower=False, include_lengths=False, pad_token=None)
        field_image = torchtext.data.Field(batch_first=True, lower=False, include_lengths=False, pad_token=None)
        field_poem = torchtext.data.Field(batch_first=True, lower=False, include_lengths=True, pad_token=None, tokenize=tokenizer)
        fields = [("id", field_id), ("url", field_url), ("image", field_image), ("poem", field_poem)]

        # Read the JSON data and download the images if they do not exist
        df = pd.read_json(filename)
        examples = []
        for _, row in tqdm(df.iterrows(), position=0, leave=True, total=len(df)):
            id = row.id
            url = row.image_url
            poem = row.poem
            # Load or download the images
            image_file = os.path.join(image_dir, f'{id}.jpg')
            try:
                image = Image.open(image_file).convert('RGB')
                if transform is not None:
                    image = transform(image)
                examples.append(torchtext.data.Example.fromlist([id, url, image, poem], fields))
            except Exception:
                pass

        super(PoemMultiMDataset, self).__init__(examples, fields)

    @classmethod
    def download(cls, root='.data', **kwargs):
        df = pd.read_json(cls.url)
        outdir = os.path.join(root, cls.dirname, cls.name)
        for _, row in tqdm(df.iterrows(), position=0, leave=True, total=len(df)):
            id = row.id
            url = row.image_url
            image_file = os.path.join(outdir, f'{id}.jpg')
            try:
                if not os.path.isfile(image_file):
                    download_image(url, image_file)
            except Exception:
                print(f"WARNING: Image {id} not downloaded from {url}.")

        return PoemMultiMDataset(cls.url, outdir, **kwargs)


class PoemUniMDatasetMasks(Dataset):
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

    def __init__(self, filename, tokenizer=None, max_seq_len=256):
        super(PoemUniMDatasetMasks, self).__init__()
        df = pd.read_json(filename)
        ids = []
        poems = []
        for _, row in tqdm(df.iterrows(), position=0, leave=True, total=len(df)):
            poems.append(row.poem)
            ids.append(row.id)
        tokens, masks = pad_bert_sequences(poems, tokenizer, max_seq_len=max_seq_len)
        self.ids = torch.tensor(ids)
        self.tokens = torch.tensor(tokens)
        self.masks = torch.tensor(masks)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.ids[index], self.tokens[index], self.masks[index]


class PoemMultiMDatasetMasks(Dataset):
    r"""MultiM Poem Dataset with masks used in the `paper <https://arxiv.org/abs/1804.08473>`__ 
    “Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training”
    from Liu, Bei et al. (2018).

    * :attr:`id` (int): Identifier of the image & poem pair.

    * :attr:`tokens` (torch.tensor): Tokenized ids of a poem.

    * :attr:`masks` (torch.tensor): Tokenized ids masked.

    * :attr:`image` (torch.tensor): Matrix of the image in RGB format.

    .. note::
        The default filename used to process the data is called ``multim_poem.json``.
        The ``image_dir`` argument is used the location of the downloaded images.
        
    .. note::
        Download the images from the csv file with the ``download`` method.

    """

    url = 'https://raw.githubusercontent.com/researchmm/img2poem/master/data/multim_poem.json'
    dirname = 'img2poem'
    name = 'multim'

    def __init__(self, filename, image_dir, tokenizer=None, max_seq_len=256, transform=None):
        super(PoemMultiMDatasetMasks, self).__init__()
        df = pd.read_json(filename)
        ids = []
        poems = []
        images = []
        for _, row in tqdm(df.iterrows(), position=0, leave=True, total=len(df)):
            id = row.id
            poem = row.poem
            image_file = os.path.join(image_dir, f'{id}.jpg')
            try:
                image = Image.open(image_file).convert('RGB')
                if transform is not None:
                    image = transform(image)
                ids.append(id)
                poems.append(poem)
                images.append(image)
            except Exception:
                pass

        tokens, masks = pad_bert_sequences(poems, tokenizer, max_seq_len=max_seq_len)
        self.ids = torch.tensor(ids)
        self.tokens = torch.tensor(tokens)
        self.masks = torch.tensor(masks)
        self.images = torch.stack(images)

    @classmethod
    def download(cls, root='.data', **kwargs):
        df = pd.read_json(cls.url)
        outdir = os.path.join(root, cls.dirname, cls.name)
        for _, row in tqdm(df.iterrows(), position=0, leave=True, total=len(df)):
            id = row.id
            url = row.image_url
            image_file = os.path.join(outdir, f'{id}.jpg')
            try:
                if not os.path.isfile(image_file):
                    download_image(url, image_file)
            except Exception:
                print(f"WARNING: Image {id} not downloaded from {url}.")

        return PoemMultiMDatasetMasks(cls.url, outdir, **kwargs)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.ids[index], self.tokens[index], self.masks[index], self.images[index]
