# File: adversarial.py
# Creation: Saturday September 19th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import sys
from tqdm import tqdm
import numpy as np
import torch

# img2poem package
from .trainer import Trainer


class AdversarialTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, **kwargs):
        super(AdversarialTrainer, self).__init__(model, optimizer, criterion, **kwargs)

    def train(self, train_loader):
        raise NotImplementedError

    def eval(self, eval_loader):
        raise NotImplementedError
