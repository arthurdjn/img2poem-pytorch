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

    from img2poem.nn import ResNet
    from img2poem.trainers import Trainer


    class ResNetTrainer(Trainer):
        def train(train_loader, criterion, optimizer):
            # train one single time the network
            pass

        def eval(eval_loader, criterion):
            # evaluate one single time the network
            pass


    model = ResNet(*args)
    trainer = ResNetTrainer(model)
    trainer.fit(epochs, train_loader, eval_loader, criterion, optimizer)

"""

# Basic imports
import os
from datetime import datetime
from abc import abstractmethod, ABC
from collections import defaultdict
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

    def __init__(self, model, optimizer, criterion, scheduler=None,
                 root="saved_models", patience=10, verbose=True, device="cpu"):
        super(Trainer, self).__init__()
        self.start = datetime.now().isoformat().split('.')[0].replace(':', '-', )
        self.root = root
        self.savedir = os.path.join(root, "saves", model.__class__.__name__, self.start)
        self.rundir = os.path.join(root, "runs", model.__class__.__name__, self.start)
        self.patience = patience
        self.verbose = verbose
        self.device = device

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.performace = defaultdict(list)
        self.tensorboard = SummaryWriter(self.rundir)
        self.early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose)

    def cuda(self):
        self.device = "cuda"
        return self

    def cpu(self):
        self.device = "cpu"
        return self

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
            train_scores = self.train(train_loader, *args, **kwargs)
            eval_scores = self.eval(eval_loader, *args, **kwargs)
            # Reduce the learning rate
            if self.scheduler is not None:
                self.scheduler.step(eval_scores["loss"])

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
            model_dict = {
                'epoch': epoch,
                'train_loss': train_scores["loss"],
                'eval_loss': eval_scores["loss"],
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'criterion': self.criterion.__class__.__name__
            }
            # Quit if early stopping
            self.early_stopping(model_dict, eval_scores["loss"].item(), epoch, self.savedir)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}...")
                return
            print()
