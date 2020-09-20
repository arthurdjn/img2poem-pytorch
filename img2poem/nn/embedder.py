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


def rank_loss(poem1, image1, poem2, image2, alpha=0.2):
    poem1 = normalize(poem1, dim=1, keepdim=True)
    poem2 = normalize(poem2, dim=1, keepdim=True)
    image1 = normalize(image1, dim=1, keepdim=True)
    image2 = normalize(image2, dim=1, keepdim=True)

    zero_tensor = torch.zeros(image1.size(0)).to(poem1.device)
    loss1 = torch.max(alpha - torch.sum(image1 * poem1, dim=1) +
                      torch.sum(image1 * poem2, dim=1), zero_tensor)
    loss2 = torch.max(alpha - torch.sum(poem2 * image2, dim=1) +
                      torch.sum(poem2 * image1, dim=1), zero_tensor)
    return torch.mean(loss1 + loss2)


class PoemEmbedder(nn.Module):
    """Model used to embed a poem in a poetic space.

    * :attr:`bert` (torch.nn.Module): BERT model used to embed poems.

    * :attr:`fc` (torch.nn.Linear): Linear layer used to map the embeddings to a poetic space.

    """

    def __init__(self, embedding_dim=512):
        super(PoemEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, embedding_dim)

    def forward(self, x, masks):
        _, out = self.bert(x, attention_mask=masks)
        return self.fc(out)

    def fine_tune(self):
        """Fine tune the BERT model by setting ``requires_grad = False`` to some learnable parameters.
        Only the classifier layer (and fully connected layer from the poem embedder)
        are learned.

        .. note::
            You should use this method before feeding the model to your training session.
            If fine tuned, the bert model has 76,900 learnable parameters.
        """
        # Do not train all embeddings
        for param in self.parameters():
            param.requires_grad = False
        # Train the classifier
        for param in self.fc.parameters():
            param.requires_grad = True

    def full_tune(self):
        """Fully tune the BERT model.

        .. warning::
            If all parameters are learnable (i.e. the model is not in fine tune mode)
            then you may face issue of memory.
            Note that by default, the bert model has 109,486,085 trainable parameters.
        """
        for param in self.parameters():
            param.requires_grad = True


class ImageEmbedder(nn.Module):
    """Model used to embed an image in a poetic space.

    * :attr:`object` (torch.nn.Module): ResNet50 model used to embed an object.

    * :attr:`sentiment` (torch.nn.Module): ResNet50 model used to embed a sentiment.

    * :attr:`scene` (torch.nn.Module): ResNet50 model used to embed a scene.

    * :attr:`fc` (torch.nn.Linear): Linear layer used to map the embeddings to a poetic space.

    """

    def __init__(self, embedding_dim=512):
        super(ImageEmbedder, self).__init__()
        self.object = ResNet50Object()
        self.sentiment = ResNet50Sentiment(num_classes=None)
        self.scene = ResNet50Scene()
        self.fc = nn.Linear(2048*3, embedding_dim)

    def forward(self, x):
        # x = B, C, H, W
        out1 = self.object(x)  # out = B, 2048
        out2 = self.sentiment(x)  # out = B, sentiment_dim
        out3 = self.scene(x)  # out = B, 2048
        # out = B, (2048 + 2048 + 2048)
        out = torch.cat([out1, out2, out3], dim=1)
        return self.fc(out)

    def from_pretrained(self, object_state=None, sentiment_state=None, scene_state=None):
        """Load weights from individual ResNet checkpoints.

        Args:
            object_state_dict ([type], optional): ResNet50Object checkpoint. Defaults to None.
            sentiment_state_dict ([type], optional): ResNet50Sentiment checkpoint. Defaults to None.
            scene_state_dict ([type], optional): ResNet50Scene checkpoint. Defaults to None.
        """
        if object_state is not None:
            checkpoint = torch.load(object_state)
            self.object.load_state_dict(checkpoint['state_dict'])
        if sentiment_state is not None:
            checkpoint = torch.load(sentiment_state)
            # Remove the classifier layer from the state dict, if exists
            checkpoint['state_dict'].pop("fc.weight", None)
            checkpoint['state_dict'].pop("fc.bias", None)
            self.sentiment.load_state_dict(checkpoint['state_dict'])
        if scene_state is not None:
            checkpoint = torch.load(scene_state)
            # Remove the classifier layer from the state dict, if exists
            checkpoint['state_dict'].pop("fc.weight", None)
            checkpoint['state_dict'].pop("fc.bias", None)
            self.scene.load_state_dict(checkpoint['state_dict'])

    def fine_tune(self):
        for param in self.object.parameters():
            param.requires_grad = False
        for param in self.sentiment.parameters():
            param.requires_grad = False
        for param in self.scene.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True

class PoeticEmbedder(nn.Module):
    """Model used to embed a pair of poem and image in a poetic space.

    * :attr:`poem_embedder` (torch.nn.Module): Poem Embedder used to map poems in a poetic space.

    * :attr:`image_embedder` (torch.nn.Module): Image Embedder used to map images in a poetic space.

    * :attr:`alpha` (float): Float used to weight poems and images.

    """

    def __init__(self, embedding_dim=512, alpha=0.2):
        super(PoeticEmbedder, self).__init__()
        self.poem_embedder = PoemEmbedder(embedding_dim=embedding_dim)
        self.image_embedder = ImageEmbedder(embedding_dim=embedding_dim)
        self.alpha = alpha

    def forward(self, poem1, mask1, image1, poem2, mask2, image2):
        poem1 = self.poem_embedder(poem1, mask1)
        poem2 = self.poem_embedder(poem2, mask2)
        image1 = self.image_embedder(image1)
        image2 = self.image_embedder(image2)
        loss = rank_loss(poem1, image1, poem2, image2, alpha=self.alpha)
        return loss, (poem1, image1, poem2, image2)

    def from_pretrained(self, **kwargs):
        self.image_embedder.from_pretrained(**kwargs)

    def fine_tune(self):
        self.poem_embedder.fine_tune()
        self.image_embedder.fine_tune()
