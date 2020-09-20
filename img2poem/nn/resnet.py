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

    """

    def __init__(self):
        super(ResNet50Object, self).__init__()
        self.backbone = resnet50(pretrained=True)
        # Do not train this model
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.size(0), -1)
        return out


class ResNet50Sentiment(nn.Module):
    """ResNet50 model to extract sentiments from an image.

    * :attr:`backbone` (torch.nn.Module): Sequential modules of the pretrained ``ResNet50`` PyTorch model.

    * :attr:`fc` (torch.nn.Module): Fully Connected layer, from ``hidden_features`` to ``out_features``.

    """

    def __init__(self, out_features=5):
        super(ResNet50Sentiment, self).__init__()
        # Fine tune this model
        self.backbone = resnet50(pretrained=True)
        self.fc = nn.Linear(2048, out_features)

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

    def __init__(self):
        super(ResNet50Scene, self).__init__()
        self.backbone = resnet50(num_classes=365)

    def forward(self, x):
        out = self.backbone(x)
        return out.view(x.size(0), -1)

    @classmethod
    def from_pretrained(cls, root='.data'):
        """Download the weights from ``Places365`` platform.

        Args:
            root (str): Saving directory.

        Returns:
            ResNet50Scene
        """
        # Download the weights
        outdir = os.path.join(root, cls.dirname, cls.name)
        weights_path = os.path.join(outdir, cls.url.split('/')[-1])
        if not os.path.exists(weights_path):
            weights_path = download_weights(cls.url, outdir)
        # Load the model from saved weights
        model = ResNet50Scene()
        checkpoint = torch.load(weights_path)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.backbone.load_state_dict(state_dict)
        # Do not train the model if pretrained
        for param in model.parameters():
            param.requires_grad = False

        return model
