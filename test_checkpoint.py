#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import reverb

import tensorflow as tf


def load_checkpoint(folder : str) -> any:
    """Try to load checkpoint from folder"""
    try:
        reader = tf.train.load_checkpoint(folder)
        shape_from_key = reader.get_variable_to_shape_map()
        dtype_from_key = reader.get_variable_to_dtype_map()

        sorted(shape_from_key.keys())
        #print(shape_from_key.keys())
        #print("--------\n")
        #print(dtype_from_key.keys())
        #print("--------\n")
        for key in shape_from_key.keys():
            #key = 'global_step/.ATTRIBUTES/VARIABLE_VALUE'
            print("Key: {} Shape: {} DType: {}".format(key, shape_from_key[key], dtype_from_key[key].name));

    except ValueError as verr:
        print("Could not load checkpoints from {1} Error: {2}".format(folder, verr))
        return None
    return folder


if __name__ == '__main__':
    folder = None

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        print("No checkpoint folder")
        exit(1)

    res = load_checkpoint(folder)
