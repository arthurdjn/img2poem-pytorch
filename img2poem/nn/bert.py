# File: bert.py
# Creation: Saturday September 12th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import torch
from pytorch_pretrained_bert import BertForMaskedLM


RoBERTa = torch.hub.load('pytorch/fairseq', 'roberta.large')

BERT = BertForMaskedLM.from_pretrained('bert-base-uncased')
