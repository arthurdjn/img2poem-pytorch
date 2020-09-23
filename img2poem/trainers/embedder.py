# File: embedder.py
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
from img2poem.nn import rank_loss


class PoeticEmbedderTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, **kwargs):
        super(PoeticEmbedderTrainer, self).__init__(model, optimizer, criterion, **kwargs)

    def train(self, train_loader):
        self.model.train()
        train_losses = []
        trange = tqdm(train_loader, desc="  Training", position=0, leave=True, total=len(train_loader), file=sys.stdout)
        for _, poem1, mask1, image1, _, poem2, mask2, image2 in trange:
            # Because gradients accumulate
            self.optimizer.zero_grad()
            # One batch prediction
            poem1 = poem1.to(self.device)
            poem2 = poem2.to(self.device)
            mask1 = mask1.to(self.device)
            mask2 = mask2.to(self.device)
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            loss, _ = self.model(poem1, mask1, image1, poem2, mask2, image2)
            train_losses.append(loss.item())
            # Compute the gradients
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=10.0)
            self.optimizer.step()
            # Print training loss
            trange.set_postfix({f"train loss": f"{loss:.6f}"})
        scores = {"loss": np.mean(train_losses)}
        return scores

    def eval(self, eval_loader):
        self.model.eval()
        eval_losses = []
        with torch.no_grad():
            trange = tqdm(eval_loader, desc="Evaluation", position=0, leave=True, total=len(eval_loader), file=sys.stdout)
            for _, poem1, mask1, image1, _, poem2, mask2, image2 in trange:
                # One eval step
                poem1 = poem1.to(self.device)
                poem2 = poem2.to(self.device)
                mask1 = mask1.to(self.device)
                mask2 = mask2.to(self.device)
                image1 = image1.to(self.device)
                image2 = image2.to(self.device)
                loss, _ = self.model(poem1, mask1, image1, poem2, mask2, image2)
                eval_losses.append(loss.item())
                # Print evaluation loss
                trange.set_postfix({f"eval loss": f"{loss:.6f}"})
        scores = {"loss": np.mean(eval_losses)}
        return scores
