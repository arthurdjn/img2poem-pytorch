# File: train_resnet50.py
# Creation: Friday September 18th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

# Move to img2poem root
import sys
sys.path.append("../")

# img2poem package
from img2poem.trainers.resnet import ResNet50SentimentTrainer
from img2poem.nn import ResNet50Sentiment
from img2poem.datasets import ImageSentimentDataset


if __name__ == "__main__":

    root = "../data/images"
    filename = "../data/images/image-Sentiment-polarity-DFE.csv"
    image_dir = "../data/images/crowdflower/sentiment"
    BATCH_SIZE = 64
    LR = 5e-5
    SPLIT = 0.9

    print(f"\n0. Hyper params...")
    print(f"\t------------------------")
    print(f"\tBatch size:       {BATCH_SIZE}")
    print(f"\tLearning Rate:    {LR}")
    print(f"\tSplit ratio:      {SPLIT}")
    print(f"\t------------------------")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    print("\n1. Loading the dataset...")
    dataset = ImageSentimentDataset(filename, image_dir, transform=transform)

    train_size = int(SPLIT * len(dataset))
    dev_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, dev_size])

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("\n2. Building the model...")
    model = ResNet50Sentiment(out_features=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=LR)
    trainer = ResNet50SentimentTrainer(model, optimizer, criterion)

    print("\n3. Training...")
    trainer.fit(train_loader, eval_loader, epochs=20, device="cpu")
