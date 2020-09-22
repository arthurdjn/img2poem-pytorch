# File: functions.py
# Creation: Saturday September 19th 2020
# Author: Arthur Dujardin
# Contact: arthur.dujardin@ensg.eu
#          arthurd@ifi.uio.no
# --------
# Copyright (c) 2020 Arthur Dujardin


def get_time(seconds):
    min, sec = divmod(seconds, 60)
    hour, min = divmod(min, 60)
    return f"{int(hour)}:{int(min)}:{int(sec)}"


def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
