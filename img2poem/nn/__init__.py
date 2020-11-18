# File: __init__.py
# Creation: Saturday September 12th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


from .adversarial import FeaturesDecoder, Discriminator
from .resnet import ResNet50Sentiment, ResNet50Object, ResNet50Scene
from .embedder import PoemEmbedder, ImageEmbedder, PoeticEmbedder
from .loss import *