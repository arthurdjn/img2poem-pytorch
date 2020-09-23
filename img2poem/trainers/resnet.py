# File: resnet.py
# Creation: Friday September 18th 2020
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
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# img2poem package
from .trainer import Trainer
from img2poem.nn.utils import one_hot


class ResNet50SentimentTrainer(Trainer):
    """Trainer for the ``ResNet50Sentiment`` model on ``ImageSentimentDataset``.

    """

    def __init__(self, model, optimizer, criterion, **kwargs):
        super(ResNet50SentimentTrainer, self).__init__(model, optimizer, criterion, **kwargs)

    def train(self, train_loader):
        self.model.train()
        train_losses = []
        labels = torch.tensor([], dtype=torch.long)
        predictions = torch.tensor([], dtype=torch.long)
        trange = tqdm(train_loader, desc="  Training", position=0, leave=True, total=len(train_loader), file=sys.stdout)
        for _, image, label in trange:
            # Because gradients accumulate
            self.optimizer.zero_grad()
            # One batch prediction
            image = image.to(self.device)  # B, C, H, W
            label = label.to(self.device)  # B, C, H, W
            pred = self.model(image)  # B, 5  -->  5 types of sentiment

            # Update the weights
            loss = self.criterion(pred, label)
            # Get the mean loss over the batch, and update the performances
            loss_avg = loss.item() / image.size(0)
            train_losses.append(loss_avg)
            # Compute the gradients
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=10.0)
            self.optimizer.step()

            # Stack the score
            prediction = torch.argmax(torch.nn.functional.softmax(pred, dim=1), dim=1)
            if label.device != "cpu":
                label = label.cpu().detach()
                prediction = prediction.cpu().detach()
            predictions = torch.cat((predictions, prediction))
            labels = torch.cat((labels, label.flatten()))

            # Print training loss
            trange.set_postfix({f"loss": f"{loss_avg:.6f}"})

        accuracy = (labels == predictions).sum().item() / len(predictions)
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        macro_f1 = f1_score(labels, predictions, average='macro')
        scores = {"loss": np.mean(train_losses), "accuracy": accuracy, "precision": precision, "recall": recall, "macro_f1": macro_f1}
        return scores

    def eval(self, eval_loader):
        self.model.eval()
        eval_losses = []
        labels = torch.tensor([], dtype=torch.long)
        predictions = torch.tensor([], dtype=torch.long)
        with torch.no_grad():
            trange = tqdm(eval_loader, desc="Evaluation", position=0, leave=True, total=len(eval_loader), file=sys.stdout)
            for _, image, label in trange:
                # One eval step
                image = image.to(self.device)  # B, C, H, W
                label = label.to(self.device)  # B, C, H, W
                pred = self.model(image)  # B, Out
                loss = self.criterion(pred, label)

                # Get the mean loss over the batch, and update the performances
                loss_avg = loss.item() / image.size(0)
                eval_losses.append(loss_avg)

                # Stack the score
                prediction = torch.argmax(torch.nn.functional.softmax(pred, dim=1), dim=1)
                if label.device != "cpu":
                    label = label.cpu().detach()
                    prediction = prediction.cpu().detach()
                predictions = torch.cat((predictions, prediction))
                labels = torch.cat((labels, label.flatten()))

                # Print evaluation loss
                trange.set_postfix({f"loss": f"{loss_avg:.6f}"})

        accuracy = (labels == predictions).sum().item() / len(predictions)
        precision = precision_score(labels, predictions, average='macro')
        recall = recall_score(labels, predictions, average='macro')
        macro_f1 = f1_score(labels, predictions, average='macro')
        scores = {"loss": np.mean(eval_losses), "accuracy": accuracy, "precision": precision, "recall": recall, "macro_f1": macro_f1}
        return scores
