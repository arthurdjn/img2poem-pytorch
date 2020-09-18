# File: sentiment.py
# Creation: Friday September 18th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin



# Basic imports
from tqdm import tqdm
import numpy as np
import torch

# AST package
from .trainer import Trainer


class ImageSentimentTrainer(Trainer):
    """Trainer for image sentiment polarity models.

    """

    def __init__(self, model, optimizer, criterion, **kwargs):
        super(ImageSentimentTrainer, self).__init__(model, optimizer, criterion, **kwargs)

    def train(self, train_iterator, device="cuda:0"):
        self.model.train()
        train_losses = []
        trange = tqdm(train_iterator, position=0, leave=True, total=len(train_iterator))
        for idx, (image, label) in trange:
            # Because gradients accumulate
            self.optimizer.zero_grad()
            image = image.to(device)    # B, C, H, W
            targets = label.to(device)  # B, C, H, W
            pred = self.model(image)    # B, Out
            loss = self.criterion(pred, targets)
            # Get the mean loss over the batch, and update the performances
            loss_avg = loss.item() / image.size(0)
            train_losses.append(loss_avg)
            # Compute the gradients
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=10.0)
            # Update weights
            self.optimizer.step()
            # Print training loss
            trange.set_postfix({f"train loss": f"{loss_avg:.6f}"})
        return torch.mean(train_losses)

    def eval(self, eval_iterator, device="cuda:0"):
        self.model.eval()
        eval_losses = []
        with torch.no_grad():
            trange = tqdm(eval_iterator, position=0, leave=True, total=len(eval_iterator))
            for idx, (inputs_idx, inputs_time, inputs), (outputs_idx, outputs_time, targets) in trange:
                # One eval step
                inputs = inputs.to(device)    # B, T, C, H, W
                targets = targets.to(device)  # B, T, C, H, W
                pred = self.model(inputs)     # B, Out
                loss = self.criterion(pred, targets)
                # Get the mean loss over the batch, and update the performances
                loss_avg = loss.item() / inputs.size(0)
                eval_losses.append(loss_avg)
                # Print evaluation loss
                trange.set_postfix({f"eval loss": f"{loss_avg:.6f}"})
        return torch.mean(eval_losses)
