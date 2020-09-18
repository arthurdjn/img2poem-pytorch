# File: image.py
# Creation: Saturday September 12th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import os
import requests
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset

# img2poem package
from io import open_files
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

    """

    url = 'https://github.com/arthurdjn/img2poem-pytorch/raw/master/data/images/image-Sentiment-polarity-DFE.csv'
    dirname = 'crowdflower'
    name = 'sentiment'
    labels2id = {
        'Highly negative': 0,
        'Negative': 1,
        'Neutral': 2,
        'Positive': 3,
        'Highly positive': 4
    }

    def __init__(self, root, transform=None):
        super(ImageSentimentDataset, self).__init__()
        data = []
        labels = []
        for file in open_files(root, ext='jpg'):
            # filename: template similar to '{sentiment}_{id}.jpg'
            sentiment = file.split('_')[0]
            image = Image.open(file).convert('RGB')
            if transform:
                image = transform(image)
            data.append(image)
            labels.append(sentiment)
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)

    @classmethod
    def download(cls, root='.data', transform=None):
        """Download the dataset from a url, and save the images to the ``root`` folder.

        Args:
            root (str, optional): Path to the saving directory. Defaults to '.data'.
            transform (torchvision.transform, optional): List of transform operations on the images. Defaults to None.

        Returns:
            ImageSentimentDataset
        """
        # Check for path issues
        outdir = os.path.join(root, cls.dirname, cls.name)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # Load the CSV data
        df = pd.read_csv(cls.url)
        for _, row in df.iterrows():
            id = row['_unit_id']
            url = row['imageurl']
            sentiment = row['which_of_these_sentiment_scores_does_the_above_image_fit_into_best']
            label = cls.labels[sentiment]
            # Download the image from the URL
            image_file = os.path.join(outdir, f'{label}_{id}.jpg')
            try:
                if not os.path.isfile(image_file):
                    download_image(url, image_file)
            except Exception as error:
                print(f"{error}. The file {id} was not downloaded from the URL {url}.")

        return ImageSentimentDataset(outdir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
