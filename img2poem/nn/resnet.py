# File: resnet.py
# Creation: Friday September 18th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import os
import torch
import torch.nn as nn
from torchvision.models import resnet50

# img2poem package
from .utils import download_weights


class ResNet50Object(nn.Module):
    """ResNet50 model to extract object elements from an image.

    * :attr:`backbone` (torch.nn.Module): Sequential modules of the pretrained ``ResNet50`` PyTorch model.

    * :attr:`fc` (torch.nn.Module): Fully Connected layer, from ``hidden_features`` to ``out_features``.

    """

    def __init__(self, hidden_features=2048, out_features=5):
        super(ResNet50Object, self).__init__()
        ResNet50 = resnet50(pretrained=True)
        modules = list(ResNet50.children())
        # Do not train this model
        for param in ResNet50.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet50Sentiment(nn.Module):
    """ResNet50 model to extract sentiments from an image.

    * :attr:`backbone` (torch.nn.Module): Sequential modules of the pretrained ``ResNet50`` PyTorch model.

    * :attr:`fc` (torch.nn.Module): Fully Connected layer, from ``hidden_features`` to ``out_features``.

    """

    def __init__(self, hidden_features=2048, out_features=5):
        super(ResNet50Sentiment, self).__init__()
        ResNet50 = resnet50(pretrained=True)
        modules = list(ResNet50.children())
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet50Scene(nn.Module):
    """ResNet50 model to extract a scene from an image.

    * :attr:`backbone` (torch.nn.Module): Sequential modules of the pretrained ``ResNet50`` PyTorch model from Place365.

    """

    url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
    dirname = 'places365'
    name = 'resnet50'

    def __init__(self, weights_path):
        super(ResNet50Scene, self).__init__()
        ResNet50 = resnet50(num_classes=365)
        ResNet50.load_state_dict(torch.load(weights_path))
        # Do not train this model
        for param in ResNet50.parameters():
            param.requires_grad = False
        modules = list(ResNet50.children())
        self.backbone = nn.Sequential(*modules)

    def forward(self, x):
        out = self.backbone(x)
        return out.view(x.size(0), -1)

    @classmethod
    def download(cls, root):
        """Download the weights from ``Places365`` platform.

        Args:
            root (str): Saving directory.

        Returns:
            ResNet50Scene
        """
        outdir = os.path.join(root, cls.dirname, cls.name)
        weights_path = download_weights(cls.url, outdir)
        return ResNet50Scene(weights_path)
