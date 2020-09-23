# File: image.py
# Creation: Saturday September 12th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import os
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

# img2poem package
from .utils import download_image


class ImageSentimentDataset(Dataset):
    """Dataset for image sentiment polarity, brought by ``crowdflower``.
    This dataset is made a cooperative dataset containing a set of images and sentiment labels, such as
    * Highly negative
    * Negative
    * Neutral
    * Positive
    * Highly positive

    * :attr:`data` (torch.tensor): Image data of all images available in the csv.

    * :attr:`labels` (torch.tensor): Image label (sentiment) for all images available in the csv.

    .. note::
        The default filename used to process the data is called ``image-Sentiment-polarity-DFE.csv``.
        The ``image_dir`` argument is used the location of the downloaded images.

    .. note::
        Download the images from the csv file with the ``download`` method.

    """

    url = 'https://github.com/arthurdjn/img2poem-pytorch/raw/master/data/images/image-Sentiment-polarity-DFE.csv'
    dirname = 'crowdflower'
    name = 'polarity'

    def __init__(self, filename, image_dir, transform=None):
        super(ImageSentimentDataset, self).__init__()
        ids = []
        images = []
        labels = []
        df = pd.read_csv(filename)
        for _, row in tqdm(df.iterrows(), desc='Loading', position=0, leave=True, total=len(df)):
            id = row['_unit_id']
            sentiment = row['which_of_these_sentiment_scores_does_the_above_image_fit_into_best']
            label = self.__class__.sentiment2label[sentiment]
            image_file = os.path.join(image_dir, f'{id}.jpg')
            try:
                image = Image.open(image_file).convert('RGB')
                if transform:
                    image = transform(image)
                ids.append(id)
                images.append(image)
                labels.append(label)
            except Exception:
                pass

        self.ids = torch.tensor(ids)
        self.images = torch.stack(images)
        self.labels = torch.tensor(labels)

    @property
    def id2label(self):
        return {
            'Highly negative': 0,
            'Negative': 1,
            'Neutral': 2,
            'Positive': 3,
            'Highly positive': 4
        }

    @property
    def label2id(self):
        return {
            0: 'Highly negative',
            1: 'Negative',
            2: 'Neutral',
            3: 'Positive',
            4: 'Highly positive'
        }

    @classmethod
    def download(cls, root='.data', **kwargs):
        """Download the dataset from a url, and save the images to the ``root`` folder.

        Args:
            root (str, optional): Path to the saving directory. Defaults to '.data'.

        Returns:
            ImageSentimentDataset
        """
        # Check for path issues
        outdir = os.path.join(root, cls.dirname, cls.name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # Load the CSV data
        df = pd.read_csv(cls.url)
        trange = tqdm(df.iterrows(), desc='Downloading', position=0, leave=True, total=len(df))
        for _, row in trange:
            id = row['_unit_id']
            url = row['imageurl']
            # Download the image from the URL
            image_file = os.path.join(outdir, f'{id}.jpg')
            try:
                if not os.path.isfile(image_file):
                    download_image(url, image_file)
            except Exception:
                print(f"WARNING: Image {id} not downloaded from {url}.")

        return ImageSentimentDataset(cls.url, outdir, **kwargs)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.ids[index], self.images[index], self.labels[index]


class ImageEmotionDataset(Dataset):
    """Dataset for image emotion polarity, brought by ``crowdflower``.
    This dataset is made a cooperative dataset containing a set of images and sentiment labels, such as
    * Highly negative
    * Negative
    * Neutral
    * Positive
    * Highly positive

    * :attr:`data` (torch.tensor): Image data of all images available in the csv.

    * :attr:`labels` (torch.tensor): Image label (emotion ids) for all images available in the csv.

    .. note::
        The default filename used to process the data is called ``image-Sentiment-emotion.csv``.
        The ``image_dir`` argument is used the location of the downloaded images.

    .. note::
        Download the images from the csv file with the ``download`` method.

    """

    url = 'https://github.com/arthurdjn/img2poem-pytorch/raw/master/data/images/image-Sentiment-emotion.csv'
    dirname = 'crowdflower'
    name = 'emotion'

    label2id = {
        'amusement': 0,
        'anger': 1,
        'awe': 2,
        'contentment': 3,
        'excitement': 4,
        'disgust': 5,
        'fear': 6,
        'sadness': 7
    }
    id2label = {
        0: 'amusement',
        1: 'anger',
        2: 'awe',
        3: 'contentment',
        4: 'excitement',
        5: 'disgust',
        6: 'fear',
        7: 'sadness'
    }

    def __init__(self, filename, image_dir, transform=None):
        super(ImageSentimentDataset, self).__init__()
        ids = []
        images = []
        labels = []
        df = pd.read_csv(filename)
        for _, row in tqdm(df.iterrows(), desc='Loading', position=0, leave=True, total=len(df)):
            id = row['id']
            emotion = row['emotion']
            label = self.__class__.sentiment2label[emotion]
            image_file = os.path.join(image_dir, f'{id}.jpg')
            try:
                image = Image.open(image_file).convert('RGB')
                if transform:
                    image = transform(image)
                ids.append(id)
                images.append(image)
                labels.append(label)
            except Exception:
                pass

        self.ids = torch.tensor(ids)
        self.images = torch.stack(images)
        self.labels = torch.tensor(labels)

    def id2label(self, id):
        return self.__class__.id2label[id]

    def label2id(self, id):
        return self.__class__.label2id[id]

    @classmethod
    def download(cls, root='.data', **kwargs):
        """Download the dataset from a url, and save the images to the ``root`` folder.

        Args:
            root (str, optional): Path to the saving directory. Defaults to '.data'.

        Returns:
            ImageSentimentDataset
        """
        # Check for path issues
        outdir = os.path.join(root, cls.dirname, cls.name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # Load the CSV data
        df = pd.read_csv(cls.url)
        trange = tqdm(df.iterrows(), desc='Downloading', position=0, leave=True, total=len(df))
        for _, row in trange:
            id = row['id']
            url = row['url']
            # Download the image from the URL
            image_file = os.path.join(outdir, f'{id}.jpg')
            try:
                if not os.path.isfile(image_file):
                    download_image(url, image_file)
            except Exception:
                print(f"WARNING: Image {id} not downloaded from {url}.")

        return ImageSentimentDataset(cls.url, outdir, **kwargs)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.ids[index], self.images[index], self.labels[index]
