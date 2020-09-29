# File: adversarial.py
# Creation: Saturday September 19th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import sys
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

# img2poem package
from .trainer import Trainer
from .earlystopping import EarlyStopping


"""
Original code from `Poem from Image repository <https://github.com/zhaoyanglijoey/Poem-From-Image>`__.
"""


class AdversarialTrainer(Trainer):
    """Train an Adversarial model architecture, 
    source code from `Pytorch documentation <https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html>`__.

    """

    def __init__(self, modelG, optimizerG, criterionG,
                 modelD, optimizerD, criterionD, schedulerD=None, root=".saved_models", **kwargs):
        super(AdversarialTrainer, self).__init__(modelG, optimizerG, criterionG, root=root, **kwargs)
        self.modelD = modelD
        self.optimizerD = optimizerD
        self.criterionD = criterionD
        self.schedulerD = schedulerD

        self.savedirD = os.path.join(root, "saves", self.modelD.__class__.__name__, self.start)
        self.rundirD = os.path.join(root, "runs", self.modelD.__class__.__name__, self.start)
        self.tensorboardD = SummaryWriter(self.rundirD)
        self.early_stoppingD = EarlyStopping(patience=self.patience, verbose=self.verbose)

    def train(self, train_loader):
        self.model.train()
        lossesD = []
        lossesR = []
        lossesG = []
        trange = tqdm(train_loader, desc="  Training", position=0, leave=True, total=len(train_loader), file=sys.stdout)
        for _, tokens, token_ids, lengths, features in trange:
            token_ids = token_ids.to(self.device)
            features = features.to(self.device)

            # 1. Update the discriminator network
            # 1.1. Real data
            self.modelD.zero_grad()
            # Do not take into accounts the <sos> token, i.e. extract tokens from index 1
            pred_real = self.modelD(token_ids[:, 1:], lengths-1)
            label_real = torch.ones(token_ids.size(0), dtype=torch.long).to(self.device)
            lossD_real = self.criterionD(pred_real, label_real)
            lossD_real.backward(retain_graph=True)
            # Get the mean loss over the batch
            lossD_real = lossD_real / features.size(0)

            # 1.2. Fake data
            pred = self.model(features, token_ids, lengths)
            # pred = B, max_seq_len
            weights = F.softmax(pred, dim=-1)
            m = Categorical(probs=weights)
            pred_token_ids = m.sample()
            # pred_token_ids = B, max_seq_len
            # Do not take into accounts the <sos> token, i.e. extract tokens from index 1
            pred_fake = self.modelD(pred_token_ids[:, 1:].detach(), lengths-1)
            label_fake = torch.zeros(token_ids.size(0), dtype=torch.long).to(self.device)
            lossD_fake = self.criterionD(pred_fake, label_fake)
            lossD_fake.backward(retain_graph=True)
            # Update the performance and weights
            lossD = lossD_real.item() + lossD_fake.item()
            lossesD.append(lossD)
            self.optimizerD.step()

            # 2. Update the generator network
            # 2.1. Train the generator (from https://pytorch.org/docs/stable/distributions.html#score-function)
            self.model.zero_grad()
            
            pred_fake = self.modelD(pred_token_ids[:, 1:].detach(), lengths-1)
            reward = F.softmax(pred_fake, dim=-1)[:, 1].unsqueeze(1)
            lossR = -m.log_prob(pred_token_ids) * reward
            lossR.sum().backward(retain_graph=True)
            lossesR.append(lossR.mean().item())

            # Measure the loss on the prediction
            label_packed = nn.utils.rnn.pack_padded_sequence(token_ids, lengths, batch_first=True)[0]
            pred_packed = nn.utils.rnn.pack_padded_sequence(pred, lengths, batch_first=True)[0]
            lossG = self.criterion(pred_packed, label_packed)
            lossG.sum().backward()
            lossesG.append(lossG.mean().item())
            # Update weights
            for param in self.model.parameters():
                torch.nn.utils.clip_grad_norm_(param, 0.25)
            self.optimizer.step()

            trange.set_postfix({f"lossD": f"{lossD:.6f}", "lossG": f"{lossG:.6f}"})

        return {"loss": np.mean(lossesD)}, {"loss": np.mean(lossesG)}

    def eval(self, eval_loader):
        self.model.train()
        lossesD = []
        lossesG = []
        lossesR = []
        trange = tqdm(eval_loader, desc="Evaluation", position=0, leave=True, total=len(eval_loader), file=sys.stdout)
        with torch.no_grad():
            for _, tokens, token_ids, lengths, features in trange:
                token_ids = token_ids.to(self.device)
                features = features.to(self.device)

                # 1. Update the discriminator network
                # 1.1. Real data
                pred_real = self.modelD(token_ids[:, 1:], lengths-1)
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
                pred_fake = self.modelD(pred_token_ids[:, 1:], lengths-1)
                label_fake = torch.zeros(token_ids.size(0), dtype=torch.long).to(self.device)
                lossD_fake = self.criterionD(pred_fake, label_fake)
                # Update the performance and weights
                lossD = lossD_real.item() + lossD_fake.item()
                lossesD.append(lossD)

                # 2. Update the generator network
                # 2.1. Train the generator (from https://pytorch.org/docs/stable/distributions.html#score-function)
                reward = F.softmax(pred_fake, dim=-1)[:, 1].unsqueeze(1)
                lossR = -m.log_prob(pred_token_ids) * reward
                lossesR.append(lossR.mean().item())

                # Measure the loss on the prediction
                label_packed = nn.utils.rnn.pack_padded_sequence(token_ids, lengths, batch_first=True)[0]
                pred_packed = nn.utils.rnn.pack_padded_sequence(pred, lengths, batch_first=True)[0]
                lossG = self.criterion(pred_packed, label_packed)
                lossesG.append(lossG.mean().item())
                trange.set_postfix({f"lossD": f"{lossD:.6f}", "lossG": f"{lossG:.6f}"})

        return {"loss": np.mean(lossesD)}, {"loss": np.mean(lossesG)}

    def sample(self, eval_loader, tokenizer, temperature=1):
        with torch.no_grad():
            for _, tokens, token_ids, lengths, features in eval_loader:
                features = features.to(self.device)
                pred_token_ids = self.model.generate(features, temperature=temperature)
                pred_token_ids = pred_token_ids[0].cpu().numpy()
                pred_tokens = tokenizer.convert_ids_to_tokens(pred_token_ids)
                pred_poem = " ".join(pred_tokens).replace(";", "\n").replace(" <pad>", "")
                poem = " ".join(tokens[0]).replace(";", "\n").replace(" <pad>", "")
                print("\n-----------------------")
                print("Poem\n")
                print(poem)
                print("\n\n")
                print("Generated\n")
                print(pred_poem)
                print("\n-----------------------\n")
                return

    def fit(self, train_loader, eval_loader, *args, epochs=10, tokenizer=None, temperature=1, sample_every=0, **kwargs):
        # Train and evaluate the model epochs times
        for epoch in range(1, epochs+1):
            epoch_len = len(str(epochs))
            print(f"Epoch: {epoch:>{epoch_len}}/{epochs}")

            # Train and evaluate the model
            train_scoresD, train_scoresG = self.train(train_loader, *args, **kwargs)
            eval_scoresD, eval_scoresG = self.eval(eval_loader, *args, **kwargs)
            # Reduce the learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            if self.schedulerD is not None:
                self.schedulerD.step()

            # Update the performances for the discriminator
            print(f"\tTrain D: {' | '.join([f'{key}: {value:.4f}' for key, value in train_scoresD.items()])}")
            print(f"\tEval  D: {' | '.join([f'{key}: {value:.4f}' for key, value in eval_scoresD.items()])}")
            for key, value in train_scoresD.items():
                self.performace[f"train_{key}"].append(value)
                self.tensorboardD.add_scalar(f'{key}/train', value, epoch)
            for key, value in eval_scoresD.items():
                self.performace[f"eval_{key}"].append(value)
                self.tensorboardD.add_scalar(f'{key}/eval', value, epoch)
            # Update the performances for the generator
            print(f"\tTrain G:   {' | '.join([f'{key}: {value:.4f}' for key, value in train_scoresG.items()])}")
            print(f"\tEval  G: {' | '.join([f'{key}: {value:.4f}' for key, value in eval_scoresG.items()])}")
            for key, value in train_scoresG.items():
                self.performace[f"train_{key}"].append(value)
                self.tensorboard.add_scalar(f'{key}/train', value, epoch)
            for key, value in eval_scoresG.items():
                self.performace[f"eval_{key}"].append(value)
                self.tensorboard.add_scalar(f'{key}/eval', value, epoch)

            # Save a checkpoint if the loss decreased
            discriminator_dict = {
                'epoch': epoch,
                'train_loss': train_scoresD["loss"],
                'eval_loss': eval_scoresD["loss"],
                'state_dict': self.modelD.state_dict(),
                'optimizer': self.optimizerD.state_dict(),
                'criterion': self.criterionD.__class__.__name__
            }
            generator_dict = {
                'epoch': epoch,
                'train_loss': train_scoresG["loss"],
                'eval_loss': eval_scoresG["loss"],
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'criterion': self.criterion.__class__.__name__
            }
            # Quit if early stopping
            self.early_stoppingD(discriminator_dict, eval_scoresD["loss"], epoch, self.savedirD)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}...")
            self.early_stopping(generator_dict, eval_scoresG["loss"], epoch, self.savedir)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}...")

            # Print the current predictions, in poem format
            if sample_every > 0 and tokenizer is not None:
                if epoch % sample_every == 0:
                    self.sample(eval_loader, tokenizer, temperature=temperature)
            print()
