# File: embedder.py
# Creation: Saturday September 12th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import torch
import torch.nn as nn
from transformers import BertModel

# img2poem package
from .resnet import ResNet50Object, ResNet50Sentiment, ResNet50Scene
from .utils import normalize


class PoemEmbedder(nn.Module):
    """Model used to embed a poem in a poetic space.

    * :attr:`bert` (torch.nn.Module): BERT model used to embed poems.

    * :attr:`fc` (torch.nn.Linear): Linear layer used to map the embeddings to a poetic space.

    """

    def __init__(self, poetic_dim=512):
        super(PoemEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, poetic_dim)

    def forward(self, x, masks):
        _, out = self.bert(x, attention_mask=masks)
        return self.fc(out)


class ImageEmbedder(nn.Module):
    """Model used to embed an image in a poetic space.

    * :attr:`object` (torch.nn.Module): ResNet50 model used to embed an object.

    * :attr:`sentiment` (torch.nn.Module): ResNet50 model used to embed a sentiment.

    * :attr:`scene` (torch.nn.Module): ResNet50 model used to embed a scene.

    * :attr:`fc` (torch.nn.Linear): Linear layer used to map the embeddings to a poetic space.

    """

    def __init__(self, sentiment_dim=5, poetic_dim=512):
        super(ImageEmbedder, self).__init__()
        self.object = ResNet50Object()
        self.sentiment = ResNet50Sentiment(sentiment_dim)
        self.scene = ResNet50Scene()
        self.fc = nn.Linear(2048*2+sentiment_dim, poetic_dim)

    def forward(self, x):
        # x = B, C, H, W
        out1 = self.object(x)  # out = B, 2048
        out2 = self.sentiment(x)  # out = B, sentiment_dim
        out3 = self.scene(x)  # out = B, 2048
        # out = B, (2048 + 2048 + sentiment_dim)
        out = torch.cat([out1, out2, out3], dim=1)
        return self.fc(out)
    
    def from_pretrained(self, object_weights=None, sentiment_weights=None, scene_weights=None):
        pass


class PoeticEmbedder(nn.Module):
    """Model used to embed a pair of poem and image in a poetic space.

    * :attr:`poem_embedder` (torch.nn.Module): Poem Embedder used to map poems in a poetic space.

    * :attr:`image_embedder` (torch.nn.Module): Image Embedder used to map images in a poetic space.

    * :attr:`alpha` (float): Float used to weight poems and images.

    """

    def __init__(self, sentiment_dim=5, poetic_dim=512, alpha=0.2):
        super(PoeticEmbedder, self).__init__()
        self.poem_embedder = PoemEmbedder(poetic_dim)
        self.image_embedder = ImageEmbedder(sentiment_dim)
        self.alpha = alpha

    def forward(self, poem1, mask1, image1, poem2, mask2, image2):
        poem_embedded1 = self.poem_embedder(poem1, mask1)
        poem_embedded2 = self.poem_embedder(poem2, mask2)
        image_embedded1 = self.img_embedder(image1)
        image_embedded2 = self.img_embedder(image2)
        return self.rank_loss(poem_embedded1, image_embedded1, poem_embedded2, image_embedded2)

    def rank_loss(self, poem_embedded1, image_embedded1, poem_embedded2, image_embedded2):
        poem_embedded1 = normalize(poem_embedded1, dim=1, keepdims=True)
        poem_embedded2 = normalize(poem_embedded2, dim=1, keepdims=True)
        image_embedded1 = normalize(image_embedded1, dim=1, keepdims=True)
        image_embedded2 = normalize(image_embedded2, dim=1, keepdims=True)

        zero_tensor = torch.zeros(image_embedded1.size(0)).to(self.device)
        loss1 = torch.max(self.alpha - torch.sum(image_embedded1 * poem_embedded1, dim=1) +
                          torch.sum(image_embedded1 * poem_embedded2, dim=1), zero_tensor)
        loss2 = torch.max(self.alpha - torch.sum(poem_embedded2 * image_embedded2, dim=1) +
                          torch.sum(poem_embedded2 * image_embedded1, dim=1), zero_tensor)
        return torch.mean(loss1 + loss2)
