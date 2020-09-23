# File: utils.py
# Creation: Friday September 18th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import os
import requests
from tqdm import tqdm
import torch


def download_weights(url, outdir='.data'):
    """Download pytorch weights from a url.

    Args:
        url (str): The url redirecting to the download link.
        outdir (str, optional): Path to the saving directory. Defaults to '.weights'.

    Returns:
        str: Path to the downloaded weights.

    Example:
        >>> url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
        >>> download_weights(url, outdir='.data')
    """
    outfile = url.split('/')[-1]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # Download the weights
    try:
        response = requests.get(url, stream=True)
        with open(os.path.join(outdir, outfile), "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)
    except Exception as error:
        print(f"{error}. Could not download the weights {outfile} from url {url}.")

    return os.path.join(outdir, outfile)


def normalize(x, **kwargs):
    r"""Normalize a tensor.

    .. math::
        \text{normalize}(t) = \frac{t}{\text{norm(t)}}

    Args:
        x (torch.tensor): Tensor to normalize.

    Returns:
        torch.tensor
    """
    out = x / torch.norm(x, **kwargs)
    return out


def one_hot(Y, num_classes):
    r"""Perform one-hot encoding on input Y.

    Args:
        Y (Tensor): 1D tensor of classes indices of length :math:`N`
        num_classes (int): number of classes :math:`C`
    
    Returns:
        Tensor: one hot encoded tensor of shape :math:`(N, C)`
    """
    batch_size = len(Y)
    Y_tilde = torch.zeros((batch_size, num_classes), device=Y.device)
    Y_tilde[torch.arange(batch_size), Y] = 1
    return Y_tilde.long()
