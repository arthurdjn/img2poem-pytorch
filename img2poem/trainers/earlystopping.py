# File: earlystopping.py
# Creation: Friday September 18th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin



r"""
This module defines an ``EarlyStopping`` object, to stop training when loss do not converge anymore.
"""


# Basic imports
import numpy as np
import torch
import os


class EarlyStopping(object):
    """
    Early stops the training if evaluation loss doesn't improve after a given patience.
    
    * :attr:`patience` (int): number of epochs to wait after last time evaluation loss improved.
    
    * :attr:`verbose` (bool): if ``True`` display log messages in the console. Default to ``False``.

    """
    def __init__(self, patience=7, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, model, eval_loss, epoch, save_path="saves"):

        score = -eval_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, eval_loss, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            print(f"Early Stopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, eval_loss, epoch, save_path)
            self.counter = 0

    def save_checkpoint(self, model, eval_loss, epoch, savedir="saves"):
        """Saves model when evaluation loss decrease.

        Args:
            eval_loss (float): evaluation's loss.
            model (torch.nn.Module): model to save.
            epoch (int): current epoch.
            savedir (str): path to the saving directory.

        """
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        if self.verbose:
            print(f"Eval loss decreased ({self.val_loss_min:.6f} --> {eval_loss:.6f})")
            print(" Saving model...")
        filepath = os.path.join(savedir, f"checkpoint_{epoch}_{eval_loss:.6f}.pth.tar")
        torch.save(model, filepath)
        self.val_loss_min = eval_loss
