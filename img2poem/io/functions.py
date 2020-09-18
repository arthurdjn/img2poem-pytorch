# File: functions.py
# Creation: Friday September 18th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


# Basic imports
import os


def open_files(dirname, ext=None, nested=True):
    """Get the path of all files from a directory.

    Args:
        dirname (string): Name or path to the directory where the files you want to open are saved..
        ext (string, optional): The files extension you want to open. The default is "geojson".

    Returns:
        list:  A list of the path of all files saved in the directory, with the extension
        geojson by default.

    Example:
        >>> dirname = "path/to/your/directory"
        >>> files = open_files(dirname, ext="jpg")
        >>> files
            ['data/image_1.jpg',
             'data/image_2.jpg',
             'data/image_3.jpg',
             'data/image_4.jpg']

    """
    try:
        ls = os.listdir(dirname)
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory {dirname} not found.")
    list_files = []
    for f in ls:
        path = os.path.join(dirname, f)
        # Look for nested folders
        if nested and os.path.isdir(path):
            sub_files = open_files(path, ext=ext, nested=nested)
            list_files.extend(sub_files)

        # Look for files
        elif os.path.isfile(path):
            if ext:
                if f.endswith(ext):
                    list_files.append(path)
            else:
                list_files.append(path)
    return list_files
