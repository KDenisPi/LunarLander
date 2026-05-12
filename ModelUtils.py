#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime
import numpy as np

def tensor_size(tnsr:any) -> any:
    if len(tnsr.shape.as_list()) == 0:
        max_min = tnsr.maximum[()] - tnsr.minimum[()]
        return max_min+1
    elif len(tnsr.shape.as_list()) == 1:
        return tnsr.shape[0]

    return tnsr.shape[0]*tnsr.shape[1]


def save_parameters(StTime, params:list, Layrs:list, Clip_layer_names:list) -> None:
    filename = "./data/parameters.csv"
    headers=['Date', 'Duration','NumIterations', 'BatchSize','UpTau', 'UpPrd', 'LrnRate', 'Gamma', 'Eps_Start', 'Eps_End', 'Eps_decay', 'GradClip', 'InitRecords']

    if os.path.exists(filename):
        headers = None

    with open(filename, 'a') as file:
        if headers:
            file.write(",".join(headers)+',')
            file.write(",".join(["LYR_{}".format(lr) for lr, _ in enumerate(Layrs)])+'\n')

        file.write("{},{},".format(datetime.now(), (datetime.now() - StTime)))
        file.write(",".join(["{:0.5f}".format(prm) for prm in params])+',')
        file.write(",".join(["{}".format(lr) for lr in Layrs]))
        file.write("," + str(Clip_layer_names)+'\n')

def param_names(q_net:any) -> list:
    val = ['Step']
    for vv in q_net.trainable_variables:
        val.append(vv.name)
    val.append('Total')
    return val

def param_gradients(step, q_net:any, grads:list, agent=None) -> None:
    val = [step]
    total_gr = 0.0

    if agent is not None and agent._grad_norm_vars is not None:
        # .numpy() works here — we are outside tf.function in eager mode
        for norm_var in agent._grad_norm_vars:
            norm = float(norm_var.numpy())
            val.append(norm)
            total_gr += norm ** 2
    else:
        # Fallback: weight norms (e.g. at step 0 before any training)
        for vv in q_net.trainable_variables:
            norm = float(np.linalg.norm(vv.numpy()))
            val.append(norm)
            total_gr += norm ** 2

    total_gr = total_gr ** 0.5
    val.append(total_gr)
    grads.append(val)

def save_results(filename:str, results:list):
    with  open(filename, 'w') as file:
        for result in results:
            file.write(str(result) + '\n')
    file.close()


def read_results(filename:str) -> list:
    results = []
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            results = file.readlines()
            results = [float(item.strip()) for item in results]
    return results

def save_info2cvs(csv_file:str, data:list, headers:list=None) -> None:
    """
    Docstring for save_info2cvs
    """
    with open(csv_file, "w") as fd_write:
        if headers:
            fd_write.write(",".join(headers)+'\n')
        for ln in data:
            fd_write.write(",".join(["{:.3f}".format(l) for l in ln])+'\n')


def save_info2list(csv_file:str, data:list, name:str) -> None:
    """
    Docstring for save_info2list
    """
    with open(csv_file, "a") as fd_write:
        fd_write.write(name + "," + str(data) +'\n')
