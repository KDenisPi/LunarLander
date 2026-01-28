#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import reverb

import tensorflow as tf

def load_checkpoint(folder : str, vars : list, summ : dict) -> any:
    """Try to load checkpoint from folder"""
    try:
        reader = tf.train.load_checkpoint(folder)
        for key in vars:
            key_short = key.replace('/.ATTRIBUTES/VARIABLE_VALUE', '')
            if key_short not in summ:
                summ[key_short] = []

            value = reader.get_tensor(key).tolist()
            lvalue = any
            if isinstance(value, type([])):
                lvalue = [round(num, 5) for num in value]
            elif isinstance(value, type(1.0)):
                lvalue = round(value, 5)
            else:
                lvalue = value

            #print("Key: {} Value: {}".format(key_short, lvalue))
            summ[key_short].append(lvalue)



    except ValueError as verr:
        print("Could not load checkpoints from {1} Error: {2}".format(folder, verr))
        return None

    return folder

def folder_ckpts_info(base_folder : str) -> dict:
    """"""
    vars = ['agent/_target_q_network/_sequential_layers/0/bias/.ATTRIBUTES/VARIABLE_VALUE',
            #'agent/_target_q_network/_sequential_layers/1/bias/.ATTRIBUTES/VARIABLE_VALUE',
            #'agent/_target_q_network/_sequential_layers/2/bias/.ATTRIBUTES/VARIABLE_VALUE',
            'agent/_target_q_network/_sequential_layers/3/bias/.ATTRIBUTES/VARIABLE_VALUE'
            #'agent/_optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE'
            ]
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1),
    )

    result = {}

    ckpt_manager = tf.train.CheckpointManager(ckpt, base_folder, max_to_keep=10)
    ckpts = ckpt_manager.checkpoints
    if ckpts is not None:
        for fld in ckpts:
            load_checkpoint(fld, vars, result)

    return result

def create_csv(data : dict, csv_prefix : str) -> None:
    """
    Docstring for create_csv
    
    :param data: Description
    :type data: dict
    """
    for key in data.keys():
        filename = csv_prefix + key.replace('/', '_') + ".csv"
        print(filename)

        with open(filename, 'w') as file:
            idx = 0
            headers = None
            for item in data[key]:
                if not isinstance(item, type([])):
                    break

                if not headers:
                    headers = ['Idx'] + [str(i) for i in range(len(item))]
                    file.write("{}\n".format(",".join(headers)))
                
                item_str = [str(i) for i in item]
                file.write("{},{}\n".format(idx, ",".join(item_str)))
                idx += 1
            file.close()

    

if __name__ == '__main__':
    folder = None
    csv_prefix = None

    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        print("No checkpoint folder\n test_checkpoints.py ckpt_folder [csv_prefix]")
        exit(1)

    if len(sys.argv) > 2:
        csv_prefix = sys.argv[2]


    res = folder_ckpts_info(folder)

    if csv_prefix:
        create_csv(res, csv_prefix)
    else:
        print(res)

