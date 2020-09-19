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


def download_weights(url, outdir='.weights'):
    """Download pytorch weights from a url.

    Args:
        url (str): The url redirecting to the download link.
        outdir (str, optional): Path to the saving directory. Defaults to '.weights'.

    Returns:
        str: Path to the downloaded weights.

    Example:
        >>> url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
        >>> download_weights(url, outdir='.weights')
    """
    # Check for path issues
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = url.split('/')[-1]
    # Download the weights
    try:
        response = requests.get(url, stream=True)
        with open(os.path.join(outdir, outfile)) as handle:
            for block in tqdm(response.iter_content(1024),
                              position=0,
                              leave=True,
                              total=len(response.iter_content(1024))):
                handle.write(block)
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
