#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import reverb

import tensorflow as tf

def load_checkpoint(folder : str, vars : list) -> any:
    """Try to load checkpoint from folder"""
    try:
        reader = tf.train.load_checkpoint(folder)
        for key in vars:
            print("Key: {} Value: {}".format(key.replace('/.ATTRIBUTES/VARIABLE_VALUE', ''), reader.get_tensor(key)))

    except ValueError as verr:
        print("Could not load checkpoints from {1} Error: {2}".format(folder, verr))
        return None
    
    return folder

def load_folders(base_folder : str) -> None:
    """
    Docstring for load_folders
    
    :param folder: Description
    :type folder: str
    """
    vars = ['agent/_target_q_network/_sequential_layers/0/bias/.ATTRIBUTES/VARIABLE_VALUE']


    f_number = 1
    folder = "{}".format(base_folder)
    while True:
        if not os.path.exists(folder):
            break
        res = load_checkpoint(folder, vars)
        if not res:
            break

        folder = "{}_{}".format(base_folder, f_number)
        f_number += 1
        print(folder)


if __name__ == '__main__':
    folder = None

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        print("No checkpoint folder")
        exit(1)

    load_folders(folder)

