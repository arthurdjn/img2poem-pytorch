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
        ResNet50 = resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        # Do not train this model
        for param in ResNet50.parameters():
            param.requires_grad = False
        self.backbone = nn.Sequential(*modules)

    def forward(self, x):
        # image = B, 3, 224, 224
        out = self.backbone(x)
        # out = B, 2048, 1, 1
        out = out.view(out.size(0), -1)
        # out = B, 2048
        return out


class ResNet50Sentiment(nn.Module):
    """ResNet50 model to extract sentiments from an image.

    * :attr:`backbone` (torch.nn.Module): Sequential modules of the pretrained ``ResNet50`` PyTorch model.

    * :attr:`fc` (torch.nn.Module): Fully Connected layer, from ``hidden_features`` to ``out_features``.

    .. note::
        This model has a classifier layer, used when the argument ``num_classes`` is not ``None``.
        This layer is used to train the model to recognize polarities from images.
        Once the model is trained, we no longer need to classify sentiments from images.
        Instead, we take the feature vector just before the last layer to feed the
        deep coupled visual-poetic model. This vector is concatenated with other visual vectors,
        like scene and object recognition.

    """

    def __init__(self, num_classes=None):
        super(ResNet50Sentiment, self).__init__()
        ResNet50 = resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        # Create a classifier layer, used when training this model only
        self.num_classes = num_classes
        if num_classes is not None:
            self.fc = nn.Linear(2048, num_classes)

    def forward(self, image):
        # image = B, 3, 224, 224
        out = self.backbone(image)
        # out = B, 2048, 1, 1
        out = out.view(out.size(0), -1)
        # out = B, 2048
        if self.num_classes is not None:
            out = self.fc(out)
            # out = B, num_classes
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
        ResNet50 = resnet50(num_classes=365)
        modules = list(ResNet50.children())[:-1]  # ignore the classifier layer
        self.backbone = nn.Sequential(*modules)

    def forward(self, x):
        # image = B, 3, 224, 224
        out = self.backbone(x)
        # out = B, 2048, 1, 1
        out = out.view(out.size(0), -1)
        # out = B, 2048
        return out

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
        checkpoint = torch.load(weights_path)
        checkpoint['state_dict'].pop('fc.weight', None)
        checkpoint['state_dict'].pop('fc.bias', None)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        resnet = resnet50(num_classes=365)
        resnet.load_state_dict(state_dict)
        modules = list(resnet.children())[:-1]  # ignore the classifier layer
        
        # Update the backbone from the ResNet Scene
        model = ResNet50Scene()
        model.backbone = nn.Sequential(*modules)
        # Do not train the model if pretrained
        for param in model.parameters():
            param.requires_grad = False

        return model
