# File: poem.py
# Creation: Saturday September 12th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import os
import pandas as pd
from torchtext.data import Dataset, Example, Field


def tokenizer(text):
    """Tokenize a string but keep ``"\n"`` character.

    Args:
        text (str): The poem in a string format.

    Returns:
        list: The words and new line characters that compose aSS poem.

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


class PoemMultiMDataset(Dataset):
    """MultiM Poem Dataset used in the paper 
    “Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training”
    from Liu, Bei et al. (2018).

    * :attr:`id` (int): Identifiant of the image & poem pair.

    * :attr:`image` (str): URL to the image.

    * :attr:`poem` (list(str)): Tokenized poem.

    """

    urls = ['https://raw.githubusercontent.com/researchmm/img2poem/master/data/multim_poem.json']
    dirname = 'researchmm'
    name = 'img2poem'

    def __init__(self, path):
        field_id = Field(batch_first=True, lower=False,
                         include_lengths=False, pad_token=None)
        field_image = Field(batch_first=True, lower=False,
                            include_lengths=False, pad_token=None)
        field_poem = Field(batch_first=True, lower=False,
                           include_lengths=True,  pad_token=None, tokenize=tokenizer)
        fields = [("id", field_id), ("image", field_image), ("poem", field_poem)]

        df = pd.read_json(path)
        examples = []
        for _, row in df.iterrows():
            id = row.id
            image = row.image_url
            poem = row.poem
            examples.append(Example.fromlist([id, image, poem], fields))
        super(PoemMultiMDataset, self).__init__(examples, fields)


class PoemUniMDataset(Dataset):
    """UniM Poem Dataset used in the paper 
    “Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training”
    from Liu, Bei et al. (2018).

    * :attr:`id` (int): Identifiant of the poem.

    * :attr:`poem` (list(str)): Tokenized poem.

    """

    urls = ['https://github.com/researchmm/img2poem/blob/master/data/unim_poem.json']
    dirname = 'researchmm'
    name = 'img2poem'

    def __init__(self, path):
        field_id = Field(batch_first=True, lower=False,
                         include_lengths=False, pad_token=None)
        field_poem = Field(batch_first=True, lower=False,
                           include_lengths=True,  pad_token=None, tokenize=tokenizer)
        fields = [("id", field_id), ("poem", field_poem)]

        df = pd.read_json(path)
        examples = []
        for _, row in df.iterrows():
            id = row.id
            poem = row.poem
            examples.append(Example.fromlist([id, poem], fields))
        super(PoemUniMDataset, self).__init__(examples, fields)

