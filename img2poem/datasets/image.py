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
import pandas as pd
from torchvision.datasets import DatasetFolder

# img2poem package
from .utils import download_image, DEFAULT_TRANSFORM


class ImagePolarityDataset(DatasetFolder):
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

    .. note::
        Processing from `PyTorch ResNet <https://pytorch.org/hub/pytorch_vision_resnet/>`__.

    """

    url = 'https://github.com/arthurdjn/img2poem-pytorch/raw/master/data/images/image-Sentiment-polarity-DFE.csv'
    dirname = 'crowdflower'
    name = 'polarity'

    id2label = {
        0: 'highly_negative',
        1: 'negative',
        2: 'neutral',
        3: 'positive',
        4: 'highly_positive'
    }
    label2id = {
        'highly_negative': 0,
        'negative': 1,
        'neutral': 2,
        'positive': 3,
        'highly_positive': 4
    }

    def __init__(self, root, transform=None):
        super(ImagePolarityDataset, self).__init__()
        transform = transform or DEFAULT_TRANSFORM
        super(ImagePolarityDataset, self).__init__(root, transform=self.transform)

    @classmethod
    def download(cls, root='.data', **kwargs):
        """Download the dataset from a url, and save the images to the ``root`` folder.

        Args:
            root (str, optional): Path to the saving directory. Defaults to '.data'.

        Returns:
            ImageSentimentDataset
        """
        # Create the main directory at the root.
        outdir = os.path.join(root, cls.dirname, cls.name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        # Create sub directories for each labels / sentiments.
        for sentiment in cls.id2label.values():
            subdir = os.path.join(outdir, sentiment)
            if not os.path.exists(subdir):
                os.makedirs(subdir)
        # Load the CSV data
        df = pd.read_csv(cls.url)
        trange = tqdm(df.iterrows(), desc='Downloading', position=0, leave=True, total=len(df))
        for _, row in trange:
            id = row['_unit_id']
            url = row['imageurl']
            sentiment = row['which_of_these_sentiment_scores_does_the_above_image_fit_into_best'].lower().replace(" ", '_')
            image_file = os.path.join(outdir, sentiment, f'{id}.jpg')
            try:
                # Download the image if it is not in its subdirectory
                if not os.path.isfile(image_file):
                    # Download the image from the URL
                    image_file = download_image(url, image_file)
                    # In case the downloaded image does not have any content (i.e. size < 1kb)
                    if os.path.getsize(image_file) < 1_000:
                        os.remove(image_file)

            except Exception as error:
                print(f"[WARNING] {error}. Image {id} not downloaded from {url}.")

        return ImagePolarityDataset(cls.url, outdir, **kwargs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        id = int(os.path.basename(path).split(".")[0])
        return id, sample, target


class ImageEmotionDataset(DatasetFolder):
    """Dataset for image emotion polarity, brought by ``crowdflower``.
    This dataset is made a cooperative dataset containing a set of images and sentiment labels, such as
    * amusement
    * anger
    * awe
    * contentment
    * excitement
    * disgust
    * fear
    * sadness

    * :attr:`data` (torch.tensor): Image data of all images available in the csv.

    * :attr:`labels` (torch.tensor): Image label (emotion ids) for all images available in the csv.

    .. note::
        The default filename used to process the data is called ``image-Sentiment-emotion.csv``.
        The ``image_dir`` argument is used the location of the downloaded images.

    .. note::
        Download the images from the csv file with the ``download`` method.

    .. note::
        Processing from `PyTorch ResNet <https://pytorch.org/hub/pytorch_vision_resnet/>`__.

    """

    url = 'https://github.com/arthurdjn/img2poem-pytorch/raw/master/data/images/image-Sentiment-emotion.csv'
    dirname = 'crowdflower'
    name = 'emotion'

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

    def __init__(self, root, transform=None):
        super(ImageEmotionDataset, self).__init__()
        transform = transform or DEFAULT_TRANSFORM
        super(ImageEmotionDataset, self).__init__(root, transform=transform)

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
            agrees = int(row['agrees'])
            disagrees = int(row['disagrees'])
            emotion = row['emotion']
            image_file = os.path.join(outdir, emotion, f'{id}.jpg')
            # Download only if people agrees on the label
            if agrees - 1 > disagrees:
                try:
                    if not os.path.isfile(image_file):
                        image_file = download_image(url, image_file)
                        # In case the downloaded image does not have any content (i.e. size < 1kb)
                        if os.path.getsize(image_file) < 1_000:
                            os.remove(image_file)
                except Exception as error:
                    print(f"[WARNING] {error}. Image {id} not downloaded from {url}.")

        return ImageEmotionDataset(root, **kwargs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        id = int(os.path.basename(path).split(".")[0])
        return id, sample, target
