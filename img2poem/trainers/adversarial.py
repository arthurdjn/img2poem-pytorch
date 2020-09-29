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
        self.modelD = discriminator
        self.optimizerD = optimizerD
        self.criterionD = criterionD

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
            pred_fake = self.modelD(pred_token_ids[:, 1:], lengths-1)
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
            pred_fake = self.modelD(pred_token_ids[:, 1:], lengths-1)
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

        return {"lossD": np.mean(lossesD),
                "lossR": np.mean(lossesR),
                "lossG": np.mean(lossesG)}

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

        return {"lossD": np.mean(lossesD),
                "lossR": np.mean(lossesR),
                "lossG": np.mean(lossesG)}

    def sample(self, eval_loader, tokenizer, temperature=1):
        with torch.no_grad():
            for _, tokens, token_ids, lengths, features in eval_loader:
                features = features.to(self.device)
                pred_token_ids = self.model.generate(features, temperature=temperature)
                pred_token_ids = pred_token_ids.cpu()
                pred_tokens = tokenizer.tokenize(pred_token_ids[0].numpy())
                pred_poem = " ".join(pred_tokens).replace(";", "\n").replace(" <pad>", "")
                print("\n-----------------------")
                print("Predicted Poem")
                print("-----------------------\n")
                print(pred_poem)
                print("\n-----------------------\n")
                return

    def fit(self, train_loader, eval_loader, *args, epochs=10, tokenizer=None, temperature=1, sample_every=0, **kwargs):
        # Train and evaluate the model epochs times
        for epoch in range(1, epochs+1):
            epoch_len = len(str(epochs))
            print(f"Epoch: {epoch:>{epoch_len}}/{epochs}")

            # Train and evaluate the model
            train_scores = self.train(train_loader, *args, **kwargs)
            eval_scores = self.eval(eval_loader, *args, **kwargs)
            # # Reduce the learning rate
            # if self.scheduler is not None:
            #     self.scheduler.step(eval_scores["loss"])

            # Update the performances
            print(f"\tTraining:   {' | '.join([f'{key}: {value:.4f}' for key, value in train_scores.items()])}")
            print(f"\tEvaluation: {' | '.join([f'{key}: {value:.4f}' for key, value in eval_scores.items()])}")
            for key, value in train_scores.items():
                self.performace[f"train_{key}"].append(value)
                self.tensorboard.add_scalar(f'{key}/train', value, epoch)
            for key, value in eval_scores.items():
                self.performace[f"eval_{key}"].append(value)
                self.tensorboard.add_scalar(f'{key}/eval', value, epoch)

            # Save a checkpoint if the loss decreased
            discriminator_dict = {
                'epoch': epoch,
                'train_loss': train_scores["lossD"],
                'eval_loss': eval_scores["lossD"],
                'state_dict': self.modelD.state_dict(),
                'optimizer': self.optimizerD.state_dict(),
                'criterion': self.criterionD.__class__.__name__
            }
            generator_dict = {
                'epoch': epoch,
                'train_loss': train_scores["lossG"],
                'eval_loss': eval_scores["lossG"],
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'criterion': self.criterion.__class__.__name__
            }
            # Quit if early stopping
            self.early_stopping(discriminator_dict, eval_scores["lossD"], epoch, self.savedir)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}...")
            self.early_stopping(generator_dict, eval_scores["lossG"], epoch, self.savedir)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}...")
            print()

            # Print the current predictions, in poem format
            if sample_every > 0 and tokenizer is not None:
                if epoch % sample_every == 0:
                    self.sample(eval_loader, tokenizer, temperature=temperature)
            print()
