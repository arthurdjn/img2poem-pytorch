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
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# img2poem package
from .trainer import Trainer


"""
Original code from `Poem from Image repository <https://github.com/zhaoyanglijoey/Poem-From-Image>`__.
"""


class AdversarialTrainer(Trainer):
    """Train an Adversarial model architecture, 
    source code from `Pytorch documentation <https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html>`__.

    """

    def __init__(self, generator, optimizerG, criterionG,
                 discriminator, optimizerD, criterionD, **kwargs):
        super(AdversarialTrainer, self).__init__(generator, optimizerG, criterionG, **kwargs)
        self.discriminator = discriminator
        self.optimizerD = optimizerD
        self.criterionD = criterionD

    def train(self, train_loader):
        self.model.train()
        lossesG = []
        lossesD = []
        trange = tqdm(train_loader, desc="  Training", position=0, leave=True, total=len(train_loader), file=sys.stdout)
        for _, features, token_ids, lengths in trange:
            # 1. Update the discriminator network
            # 1.1. Real data
            self.discriminator.zero_grad()
            pred_real = self.discriminator(token_ids, lengths)
            label_real = torch.ones(token_ids.size(0), dtype=torch.long).to(self.device)
            lossD_real = self.criterionD(pred_real, label_real)
            lossD_real.backward()
            # Get the mean loss over the batch
            lossD_real = lossD_real / features.size(0)

            # 1.2. Fake data
            pred = self.model(features, token_ids, lengths)
            # pred = B, max_seq_len
            weights = F.softmax(pred, dim=-1)
            m = Categorical(probs=weights)
            pred_token_ids = m.sample()
            # pred_token_ids = B, max_seq_len
            pred_fake = self.discriminator(pred_token_ids.detach(), lengths)
            label_fake = torch.zeros(token_ids.size(0), dtype=torch.long).to(self.device)
            lossD_fake = self.criterionD(pred_fake, label_fake)
            lossD_fake.backward()
            # Update the performance and weights
            lossD = lossD_real.item() + lossD_fake.item()
            lossesD.append(lossD)
            self.optimizerD.step()

            # 2. Update the generator network
            # 2.1. Train the generator (from https://pytorch.org/docs/stable/distributions.html#score-function)
            self.model.zero_grad()
            reward = F.softmax(pred_fake, dim=-1)[:, 1].unsqueeze(-1)
            lossR = -m.log_prob(pred_token_ids) * reward
            lossR.backward()
            lossR = lossR.mean().item()
            
            # Measure the loss on the prediction
            labels_pack, _ = nn.utils.rnn.pack_padded_sequence(token_ids[:, 1:], lengths, batch_first=True)
            pred_pack = nn.utils.rnn.pack_padded_sequence(pred, lengths, batch_first=True)
            lossG = self.criterion(pred_pack, labels_pack)
            lossG.backward()
            lossG = lossG.mean().item()
            lossesG.append(lossG)

    def eval(self, eval_loader):
        self.model.eval()
        lossesG = []
        lossesD = []
        trange = tqdm(eval_loader, desc="Evaluation", position=0, leave=True, total=len(eval_loader), file=sys.stdout)
        with torch.no_grad():
            for _, features, token_ids, lengths in trange:
                # 1. Update the discriminator network
                # 1.1. Real data
                pred_real = self.discriminator(token_ids, lengths)
                label_real = torch.ones(token_ids.size(0), dtype=torch.long).to(self.device)
                lossD_real = self.criterionD(pred_real, label_real)
                # Get the mean loss over the batch
                lossD_real = lossD_real / features.size(0)

                # 1.2. Fake data
                pred = self.model(features, token_ids, lengths)
                # pred = B, max_seq_len
                weights = F.softmax(pred, dim=-1)
                m = Categorical(probs=weights)
                pred_token_ids = m.sample()
                # pred_token_ids = B, max_seq_len
                pred_fake = self.discriminator(pred_token_ids.detach(), lengths)
                label_fake = torch.zeros(token_ids.size(0)).long().to(self.device)
                lossD_fake = self.criterionD(pred_fake, label_fake)
                # Update the performance and weights
                lossD = lossD_real.item() + lossD_fake.item()
                lossesD.append(lossD)

                # 2. Update the generator network
                # 2.1. Train the generator (from https://pytorch.org/docs/stable/distributions.html#score-function)
                reward = F.softmax(pred_fake, dim=-1)[:, 1].unsqueeze(-1)
                lossR = -m.log_prob(pred_token_ids) * reward
                lossR = lossR.mean().item()
                
                # Measure the loss on the prediction
                labels_pack, _ = nn.utils.rnn.pack_padded_sequence(token_ids[:, 1:], lengths, batch_first=True)
                pred_pack = nn.utils.rnn.pack_padded_sequence(pred, lengths, batch_first=True)
                lossG = self.criterion(pred_pack, labels_pack)
                lossG = lossG.mean().item()
                lossesG.append(lossG)
