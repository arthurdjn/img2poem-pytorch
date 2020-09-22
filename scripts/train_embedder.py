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
from img2poem.utils import count_parameters


if __name__ == "__main__":

    ROOT = "../"
    FILENAME = f"{ROOT}/data/images/image-Sentiment-polarity-DFE.csv"
    IMAGE_DIR = f"{ROOT}/data/images/crowdflower/sentiment"
    
    RESNET_SENTIMENT_STATE = f'{ROOT}/models/resnet50_sentiment.pth.tar'
    RESNET_SCENE_STATE = f'{ROOT}/models/resnet50_scene.pth.tar'

    BATCH_SIZE = 32
    LR = 1e-4
    SPLIT = 0.9

    print(f"\n0. Hyper params...")
    print(f"\t------------------------")
    print(f"\tBatch size:       {BATCH_SIZE}")
    print(f"\tLearning Rate:    {LR}")
    print(f"\tSplit ratio:      {SPLIT}")
    print(f"\t------------------------")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_seq_len = tokenizer.max_model_input_sizes['bert-base-uncased']
    max_seq_len = 128

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    print("\n1. Load the dataset...")
    dataset = PoeticEmbeddedDataset(FILENAME, IMAGE_DIR,
                                    tokenizer=tokenizer,
                                    max_seq_len=max_seq_len,
                                    transform=transform)

    print(f"Dataset ids size: {dataset.ids.shape}")
    print(f"Dataset images size: {dataset.images.shape}")
    print(f"Dataset token ids size: {dataset.token_ids.shape}")
    
    train_size = int(SPLIT * len(dataset))
    dev_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, dev_size])

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("\n2. Build the model...")
    model = PoeticEmbedder(embedding_dim=512, alpha=0.2)
    model.from_pretrained(sentiment_state=RESNET_SENTIMENT_STATE, 
                          scene_state=RESNET_SCENE_STATE)
    model.fine_tune()
    
    print(f'The model has {count_parameters(model):,} trainable parameters')

    print("\n3. Build the trainer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=LR)
    trainer = PoeticEmbedderTrainer(model, optimizer, criterion)

    print("\n4. Train...")
    trainer.fit(train_loader, eval_loader, epochs=100)
