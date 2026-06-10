#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import ModelUtils as mutils

results = mutils.read_list2info('data/results.csv')
results_n = mutils.info2n_list(results)

print_prms = False
for cmd in sys.argv:
    if cmd.find("--params") >= 0:
        print_prms = True

selected = []

for res in results_n:
    #print(res)
    sum_last = sum(res[-5:]) if len(res) > 6  else sum(res[1:])
    if sum_last > 0:
        print("{} {:.5f} {}".format(res[0], sum_last, res[-5:]))
        selected.append(res[0])

#print(selected)

if print_prms:
    params = mutils.read_list2info('data/parameters.csv')
    for prm in params:
        idx = prm.split(",")[1]
        if idx in selected:
            print(prm, end="")
