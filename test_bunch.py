#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from datetime import datetime

import LunarLander as llun

initial_cfg = {
    "replay_buffer_max_length": 100000,
    "num_eval_episodes": 10,
    "num_iterations": 100000,
    "eval_interval": 5000,
    "log_interval": 5000,
    "batch_size": 128,

    "debug" : False,
    "debug_data" : "data/",
    "generate_cvs": True,


    "optimizer": 0.001,
    "gamma": 0.9,

    "layers": [
        {
            "num_units_scale": 1,
            "layer_params": 100,
            "activation": "relu"
        },
        {
            "num_units_scale": 1,
            "layer_params": 50,
            "activation": "relu"
        }
    ]
}


#tres = [(0, -1000), (100, -1000), (1200, -4560)]

cfg = llun.ModelParams()
with open("data/results.log", "a") as fd_write:
    for optimizer, gamma in [(0.001, 0.9), (0.01, 0.9), (0.01, 0.99), (0.001, 0.99)]:
        #print("Opt: {} Gamma: {}".format(optimizer, gamma))
        for units_scale in [1,2]:
            #print("num_units_scale: {}".format(units_scale))

            tm = datetime.now()
            model_name = "{}-{}-{}_{}_{}{}{}".format(tm.year, tm.month, tm.day, tm.hour, optimizer, gamma, units_scale)

            print("{} {}".format(tm, model_name))

            json = initial_cfg
            json["optimizer"] = optimizer
            json["gamma"] = gamma

            json["layers"][0]["num_units_scale"] = units_scale
            json["layers"][1]["num_units_scale"] = units_scale

            cfg.parse_json(json)
            ll = llun.LunarLander(cfg, model_name=model_name)
            ll.prepare()
            tres = ll.train_agent()

            fd_write.write(model_name+',')
            fd_write.write(",".join(str(item) for item in tres)+'\n')
    fd_write.close()
