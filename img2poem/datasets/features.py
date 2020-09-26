# File: features.py
# Creation: Saturday September 26th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# img2poem package
from .utils import download_image, default_transform
from img2poem.tokenizer import pad_bert_sequences