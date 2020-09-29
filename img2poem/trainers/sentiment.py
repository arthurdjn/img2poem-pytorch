# File: sentiment.py
# Creation: Friday September 18th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
from tqdm import tqdm
import torch

# img2poem package
from .trainer import Trainer


class ResNet50SentimentTrainer(Trainer):
    """Trainer for the ``ResNet50Sentiment`` model on ``ImageSentimentDataset``.

    """

    def __init__(self, model, optimizer, criterion, **kwargs):
        super(ResNet50SentimentTrainer, self).__init__(model, optimizer, criterion, **kwargs)

    def train(self, train_loader, device="cuda:0"):
        self.model.train()
        train_losses = []
        trange = tqdm(train_loader, position=0, leave=True, total=len(train_loader))
        for _, (image, label) in trange:
            # Because gradients accumulate
            self.optimizer.zero_grad()
            # One batch prediction
            image = image.to(device)  # B, C, H, W
            label = label.to(device)  # B, C, H, W
            pred = self.model(image)  # B, Out
            loss = self.criterion(pred, label)
            # Get the mean loss over the batch, and update the performances
            loss_avg = loss.item() / image.size(0)
            train_losses.append(loss_avg)
            # Compute the gradients
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=10.0)
            self.optimizer.step()
            # Print training loss
            trange.set_postfix({f"train loss": f"{loss_avg:.6f}"})
        return torch.mean(train_losses)

    def eval(self, eval_loader, device="cuda:0"):
        self.model.eval()
        eval_losses = []
        with torch.no_grad():
            trange = tqdm(eval_loader, position=0, leave=True, total=len(eval_loader))
            for _, (image, label) in trange:
                # One eval step
                image = image.to(device)  # B, C, H, W
                label = label.to(device)  # B, C, H, W
                pred = self.model(image)  # B, Out
                loss = self.criterion(pred, label)
                # Get the mean loss over the batch, and update the performances
                loss_avg = loss.item() / image.size(0)
                eval_losses.append(loss_avg)
                # Print evaluation loss
                trange.set_postfix({f"eval loss": f"{loss_avg:.6f}"})
        return torch.mean(eval_losses)
