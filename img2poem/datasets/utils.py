# File: utils.py
# Creation: Saturday September 12th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import requests
import os
import torchvision.transforms as transforms


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # To float values between [0, 1]
    transforms.ToTensor(),
    # Normalize regarding ResNet training data
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


def download_image(url, outname, outdir='./'):
    """Download an image from a URL.
    Original code from a `post <https://stackoverflow.com/a/30229298>`__ on Stackoverflow.

    Args:
        url (str): URL to the image.
        outname (str): Name of the image, with its extension.
        outdir (str): Path to the saving directory.

    Example:
        >>> url = "http://farm4.staticflickr.com/3910/14393814286_c6dbcf7a92_z.jpg"
        >>> outname = "my_image.png"
        >>> download_image(url, outname)
    """
    with open(os.path.join(outdir, outname), 'wb') as handle:
        try:
            response = requests.get(url, stream=True)
            for block in response.iter_content(1024):
                handle.write(block)
        except Exception as error:
            print(f"WARNING: An error occured. {error}"
                  f"Could not download the image {outdir}/{outname} from the URL {url}.")
