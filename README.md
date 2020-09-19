![Python](https://img.shields.io/badge/python-3.+-blue.svg)
![pytorch](https://img.shields.io/badge/pytorch-1.6.0-ee4c2c.svg)
[![arxiv](https://img.shields.io/badge/paper-arxiv-red.svg)](https://arxiv.org/abs/1804.08473)


<h1>img2poem-pytorch 🖼️ 📃</h1>

> _PyTorch implementation of the paper ‟Beyond Narrative Description: Generating Poetry from Images” by B. Liu et al., 2018._

<h3>Currently in progress ! 💻</h3>

_Feel free to star the project or create an issue_

<h1>Table of Contents</h1>

<!-- TOC -->

- [1. Overview](#1-overview)
    - [1.1. Get Started](#11-get-started)
- [2. Datasets](#2-datasets)
    - [2.1. Downloads](#21-downloads)
- [3. Architecture](#3-architecture)
    - [3.1. Training](#31-training)
        - [3.1.1. Image](#311-image)
        - [3.1.2. Poetic Alignment](#312-poetic-alignment)
        - [3.1.3. Generator](#313-generator)
- [4. References](#4-references)

<!-- /TOC -->

# 1. Overview

This project introduces poem generation from images. This implementation was inspired from the research paper [‟Beyond Narrative Description: Generating Poetry from Images” by Liu, Bei et al.](https://arxiv.org/abs/1804.08473), published in 2018 at Microsoft.

The implementation is already [coded with TensorFlow](https://github.com/researchmm/img2poem) in the official [Microsoft](https://github.com/researchmm) repository. Even though a similar implementation has been transcribed in PyTorch from the project [“Neural Poetry Generation with Visual Inspiration.” by Li, Zhaoyang et al.](https://github.com/zhaoyanglijoey/Poem-From-Image), their implementation does not totally match the original paper.

This repository tries to rearrange code from [Zhaoyang Li et al.](https://github.com/zhaoyanglijoey/Poem-From-Image) implementation and create a model architecture similar to [Bei, Liu et al.](https://github.com/researchmm/img2poem), with PyTorch.

## 1.1. Get Started

To use this project, clone the repository from the command line with:

```bash
$ git clone https://github.com/arthurdjn/img2poem-pytorch
```

Then, navigate to the project root:

```bash
$ cd img2poem-pytorch
```

# 2. Datasets

To train the models, you will need to download the datasets used in this project.

The datasets used are:
* ``PoemUniMDatasetMasked``: a dataset of poems only,
* ``PoemMuliMDatasetMasked``: a dataset of paired poems and images,
* ``PoeticEmbeddedDataset``: a dataset to align poems and images.
* ``ImageSentimentDataset``: a dataset of images and polarities,

## 2.1. Downloads

To download the dataset, use the ``download()`` method, defined for all datasets.
It will downloads poems and images in a ``root`` folder.

For example, you can use:

```python
from img2poem.datasets import ImageSentimentDataset

dataset = ImageSentimentDataset.download(root='.data')
```

# 3. Architecture

The architecture is decomposed in two parts:
* Encoder, used to extract poeticness from an image,
* Decoder, used to generate a poem from a poetic space.

The encoder is made of three CNN, used to extract scene, object, and sentiment information.
To align these features in a poetic space, this encoder is used with a BERT model, to align visual feature with their paired poems.

Then, the decoder works with a discriminator which evaluates the poeticness of a generated poem.

## 3.1. Training

### 3.1.1. Image

The visual encoder is made of three CNN.

#### 3.1.1. ResNet50 Object

The object detection classifier is the vanilla ``ResNet50``, from TorchVision. More info [here](https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet50).

#### 3.1.2. ResNet50 Scenes

The scene classifier is a ``ResNet50`` model fine tuned on the Places365 dataset.
You can find the weights on the MIT platform [here](http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar).

#### 3.1.3. ResNet50 Sentiment

To train the visual sentiment classifier, use the ``ImageSentimentDataset`` with the ``ResNet50Sentiment`` model.

You can use the script ``scripts/train_resnet50.py`` to fine tune the model:

```bash
$ python scripts/train_resnet50.py
```

```bash
Hyper params...
------------------------
Batch size:       64
Learning Rate:    5e-05
Split ratio:      0.9
------------------------

Loading the dataset...
100%|█████████████████████████████████| 15613/15613 [01:16<00:00, 203.41it/s]

Building the model...
done

Training...
Epoch 12/100 14%|█████                           | 7/199 [01:16<00:00, 203.41it/s, train loss=0.01034]
...
```

### 3.1.2. Poetic Alignment

### 3.1.3. Generator

# 4. References

- [1] Liu, Bei et al. “Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training”, 2018. _ACM Multimedia Conference - ACM MM2018._  
[Paper](https://arxiv.org/abs/1804.08473) | [GitHub](https://github.com/researchmm/img2poem)

- [2] Li, Zhaoyang et al. “Neural Poetry Generation with Visual Inspiration.”, 2018.  
[Paper](https://github.com/zhaoyanglijoey/Poem-From-Image/blob/master/419_PoemGen_Report.pdf) | [GitHub](https://github.com/zhaoyanglijoey/Poem-From-Image)
