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
from torchtext.data import Dataset, Example, Field

# img2poem package
from .utils import download_image


def tokenizer(text):
    r"""Tokenize a string but keeps ``"\n"`` characters.

    Args:
        text (str): The poem in a string format.

    Returns:
        list: The words and new line characters that compose a poem.

    Example:
        >>> poem = "what is lovely never dies\nbut passes into other loveliness\nstar-dust or sea-foam flower or winged air"
        >>> tokenizer(poem)
            ['what', 'is', 'lovely', 'never', 'dies', '\n', 
            'but', 'passes', 'into', 'other', 'loveliness', '\n', 
            'star-dust', 'or', 'sea-foam', 'flower', 'or', 'winged', 'air']
    """
    lines = text.splitlines(keepends=True)
    splits = []
    for i, line in enumerate(lines):
        splits.extend(line.split())
        if i < len(lines) - 1:
            splits.append("\n")
    return splits


class PoemUniMDataset(Dataset):
    r"""UniM Poem Dataset used in the `paper <https://arxiv.org/abs/1804.08473>`__ 
    “Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training”
    from Liu, Bei et al. (2018).

    * :attr:`id` (int): Identifier of the poem.

    * :attr:`poem` (list(str)): Tokenized poem.

    """

    urls = ['https://github.com/researchmm/img2poem/blob/master/data/unim_poem.json']
    dirname = 'img2poem'
    name = 'unim'

    def __init__(self, path):
        # Define fields that compose an example
        field_id = Field(batch_first=True, lower=False, include_lengths=False, pad_token=None)
        field_poem = Field(batch_first=True, lower=False, include_lengths=True, pad_token=None, tokenize=tokenizer)
        fields = [("id", field_id), ("poem", field_poem)]

        df = pd.read_json(path)
        examples = []
        for _, row in df.iterrows():
            examples.append(Example.fromlist([row.id, row.poem], fields))
        super(PoemUniMDataset, self).__init__(examples, fields)


class PoemMultiMDataset(Dataset):
    r"""MultiM Poem Dataset used in the `paper <https://arxiv.org/abs/1804.08473>`__ 
    “Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training”
    from Liu, Bei et al. (2018).

    * :attr:`id` (int): Identifier of the image & poem pair.

    * :attr:`url` (str): URL to the image.

    * :attr:`image` (np.ndarray): Matrix of the image in RGB format.

    * :attr:`poem` (list(str)): Tokenized poem.

    """

    urls = ['https://raw.githubusercontent.com/researchmm/img2poem/master/data/multim_poem.json']
    dirname = 'img2poem'
    name = 'multim'

    def __init__(self, path, tokenizer=None):
        # Define fields that compose an example
        field_id = Field(batch_first=True, lower=False, include_lengths=False, pad_token=None)
        field_url = Field(batch_first=True, lower=False, include_lengths=False, pad_token=None)
        field_image = Field(batch_first=True, lower=False, include_lengths=False, pad_token=None)
        field_poem = Field(batch_first=True, lower=False, include_lengths=True, pad_token=None, tokenize=tokenizer)
        fields = [("id", field_id), ("url", field_url), ("image", field_image), ("poem", field_poem)]

        # Check for directories
        dirname = os.path.dirname(path)
        dirimages = os.path.join(dirname, 'multim_images')
        if not os.path.exists(dirimages):
            os.makedirs(dirimages)
        
        # Read the JSON data and download the images if they do not exist
        df = pd.read_json(path)
        examples = []
        for _, row in tqdm(df.iterrows()):
            id = row.id
            url = row.image_url
            poem = row.poem
            # Load or download the images
            image_file = os.path.join(dirimages, f'{id}.jpg')
            try:
                if not os.path.isfile(image_file):
                    download_image(url, image_file)
                image = Image.open(image_file).convert('RGB')
                # Create an example if the image is correctly loaded
                examples.append(Example.fromlist([id, url, image, poem], fields))
            except Exception as error:
                print(f"{error}. The file {id} was not downloaded from the URL {url}.")

        super(PoemMultiMDataset, self).__init__(examples, fields)
