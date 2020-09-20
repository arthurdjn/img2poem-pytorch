# File: train_embedder.py
# Creation: Saturday September 19th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer

# img2poem package
from img2poem.trainers.embedder import PoeticEmbedderTrainer
from img2poem.nn import PoeticEmbedder
from img2poem.datasets import PoeticEmbeddedDataset


if __name__ == "__main__":

    ROOT = "../data/images"
    FILENAME = "../data/images/image-Sentiment-polarity-DFE.csv"
    IMAGE_DIR = "../data/images/crowdflower/sentiment"
    RESNET_SENTIMENT_STATE = "../models/resnet/resnet50_sentiment.pth.tar"
    BATCH_SIZE = 64
    LR = 5e-5
    SPLIT = 0.9

    print(f"\n0. Hyper params...")
    print(f"\t------------------------")
    print(f"\tBatch size:       {BATCH_SIZE}")
    print(f"\tLearning Rate:    {LR}")
    print(f"\tSplit ratio:      {SPLIT}")
    print(f"\t------------------------")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    print("\n1. Loading the dataset...")
    dataset = PoeticEmbeddedDataset(FILENAME, IMAGE_DIR, transform=transform)

    train_size = int(SPLIT * len(dataset))
    dev_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, dev_size])

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("\n2. Building the model...")
    model = PoeticEmbedder(sentiment_dim=5, embedding_dim=512, alpha=0.2)
    model.sentiment.load_state_dict(torch.load(RESNET_SENTIMENT_STATE))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=LR)
    trainer = PoeticEmbedderTrainer(model, optimizer, criterion)

    print("\n3. Training...")
    trainer.fit(train_loader, eval_loader, epochs=20)