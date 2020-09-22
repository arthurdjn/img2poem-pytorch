# File: loss.py
# Creation: Tuesday September 22nd 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import torch

# img2poem package
from .utils import normalize


def rank_loss(poem1, image1, poem2, image2, alpha=0.2):
    """Rank loss described in the paper by `Liu, Bei et al.
     “Beyond Narrative Description.” 2018 ACM Multimedia Conference <https://arxiv.org/pdf/1804.08473.pdf>`__.

    Args:
        poem1 (torch.tensor): First poem tensor of shape :math:`(B, F)`.
        image1 (torch.tensor): Image associated to the first poem, of shape :math:`(B, F)`.
        poem2 (torch.tensor): Second poem tensor of shape :math:`(B, F)`.
        image2 (torch.tensor): Image associated to the second poem, of shape :math:`(B, F)`.
        alpha (float, optional): Regularizer. Defaults to 0.2.

    Returns:
        Loss
    """
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
