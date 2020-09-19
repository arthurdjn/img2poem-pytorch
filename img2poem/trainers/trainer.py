# File: trainer.py
# Creation: Friday September 18th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin



r"""
A ``Trainer`` is used to wrap a `PyTorch` ``Module`` to provide inner methods to train, evaluate and fit a model 
just like for `TensorFlow` models.

The `API` is similar to `sklearn` or `TensorFlow`.

.. code-block:: python

    from astnets.torch.nn import UNet
    from astnets.torch import Trainer


    class UNetTrainer(Trainer):
        def train(train_loader, criterion, optimizer):
            # train one single time the network
            pass

        def eval(eval_loader, criterion):
            # evaluate one single time the network
            pass


    model = UNet(*args)
    trainer = UNetTrainer(model)
    trainer.fit(epochs, train_loader, eval_loader, criterion, optimizer)

"""

# Basic imports
import os
from datetime import datetime
from abc import abstractmethod, ABC
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

# img2poem package
from .earlystopping import EarlyStopping


class Trainer(ABC):
    r"""
    This template is easier to handle for hyperparameters optimization, as the ``fit``, ``train``, ``evaluate``
    methods are part of the model.

    * :attr:`model` (torch.nn.Module): `PyTorch` module to train.

    * :attr:`optimizer` (torch.optim.Optimizer): optimizer for weights and biases.

    * :attr:`criterion` (torch.nn.Loss): loss function.

    * :attr:`performance` (dict): dict containing the different metrics over the train and eval sessions.

    * :attr:`tensorboard` (torch.utils.tensorboard.SummaryWriter): `TensorBoard` to save intermediate results and states.

    * :attr:`rundir` (str): saving directory for tensorboard runs.

    * :attr:`savedir` (str): saving directory for models.

    * :attr:`patience` (int): number of epochs to wait after last time evaluation loss improved.
    
    * :attr:`verbose` (bool): if ``True`` display log messages in the console. Default to ``False``.

    """

    def __init__(self, model, optimizer, criterion, 
                 rundir="runs", savedir="saves", patience=7, verbose=True):
        super(Trainer, self).__init__()
        self.start = datetime.now().isoformat().split('.')[0].replace(':', '-', )
        self.savedir = os.path.join(savedir, model.__class__.__name__, self.start)
        self._rundir = os.path.join(rundir, model.__class__.__name__, self.start)
        self._patience = patience
        self._verbose = verbose
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.performace = defaultdict(list)
        self.tensorboard = SummaryWriter(self.rundir)
        self.early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose)

    @property
    def verbose(self):
        return self._verbose

    @property
    def patience(self):
        return self._patience

    @property
    def rundir(self):
        return self._rundir

    @verbose.setter
    def verbose(self, value):
        self.early_stopping = EarlyStopping(self.patience, value)

    @patience.setter
    def patience(self, value):
        self.early_stopping = EarlyStopping(value, self.verbose)

    @rundir.setter
    def rundir(self, value):
        self.tensorboard = SummaryWriter(value)

    @abstractmethod
    def train(self, train_loader, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def eval(self, eval_loader, *args, **kwargs):
        raise NotImplementedError

    def fit(self, train_loader, eval_loader, *args, epochs=10, **kwargs):
        # Train and evaluate the model epochs times
        for epoch in range(1, epochs+1):
            epoch_len = len(str(epochs))
            print(f"Epoch: {epoch:>{epoch_len}}/{epochs}")

            # Train and evaluate the model
            train_loss = self.train(train_loader, *args, **kwargs)
            eval_loss = self.eval(eval_loader, *args, **kwargs)
            self.performace["train_loss"].append(train_loss)
            self.performace["eval_loss"].append(eval_loss)
            print(f"\tTraining:   loss={train_loss:.6f}")
            print(f"\tEvaluation: loss={eval_loss:.6f}")
            self.tensorboard.add_scalar('Loss/train', train_loss, epoch)
            self.tensorboard.add_scalar('Loss/eval', eval_loss, epoch)
            # Fix LR
            pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              factor=0.5,
                                                              patience=self.patience,
                                                              verbose=self.verbose)
            pla_lr_scheduler.step(eval_loss)  # lr_scheduler
            
            # Save a checkpoint
            model_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'criterion': self.criterion.__class__.__name__
            }
            self.early_stopping(model_dict, eval_loss.item(), epoch, self.savedir)
            # Quit if early stopping
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}...")
                return
