#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import base64
from datetime import datetime
import signal
import math

import numpy as np
import reverb

import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout

from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import tensor_spec
from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.networks import sequential
from tf_agents.utils import common
from tf_agents.policies import py_tf_eager_policy, random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.drivers import py_driver
from tf_agents.networks.layer_utils import print_summary


tf.compat.v1.enable_v2_behavior()

#env_name = 'LunarLander-v2' # @param {type:"string"}
env_name='CartPole-v1'

num_iterations = 80000 if env_name == 'LunarLander-v2' else 80000
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = num_iterations*2 if num_iterations <= 120000 else num_iterations + 50000 # @param {type:"integer"}
num_initial_records = 15000 #if num_iterations <= 100000 else 5000 #1000
refill_buffer_interval=0

batch_size = 256 # 256 #256

train_driver_max_step=1

num_eval_episodes = 10 # @param {type:"integer"}

eval_interval = 20000 # 100 @param {type:"integer"}
log_interval = 5000 # 50 @param {type:"integer"}
log_loss_interval = log_interval

#checkpoints
ckpt_max_to_keep = 20
episode_for_checkpoint = eval_interval

flush_interval = 0 #5000

target_update_tau=0.001 #0.005 #0.05	    #Factor for soft update of the target networks.
target_update_period=30 #10 #5 	    #Period for soft update of the target networks.

#layer_sz = [128, 128, 64]
layer_sz = [128, 128] #[128, 256] #[128, 64]

#In reinforcement learning (RL) and analysis, bias refers to
#the systematic error or difference between an agent’s predicted value (reward) and the true, actual value.
#High bias means the model makes overly simplified assumptions, failing to capture the true reward structure,
#which can cause the agent to learn incorrect or sub-optimal policies.

bias = [tf.keras.initializers.Constant(0.0)] * len(layer_sz) #-0.2
dropout = [0.0] * len(layer_sz) if env_name=='LunarLander-v2' else [0.0] * len(layer_sz)
#dropout[-1] = 0.5


bias_lyr_out = tf.keras.initializers.Constant(0)

kernel_init = [
            tf.keras.initializers.VarianceScaling(
                scale=1.0,
                mode='fan_in',
                distribution='truncated_normal')] * len(layer_sz)

kernel_init_lyr_out = tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03)

#
#Important parameters
# lerning rate - Adam optimizer parameter (0.001 - 0.0001)
# the discount factor (γ) of future rewards - gamma (0.9-1.0)
#
#Epsilon-Greedy Exploration: A strategy used during training to balance exploration
#(taking random actions to discover new possibilities) and exploitation (taking the action with the highest predicted Q-value).
#The probability of taking a random action, epsilon, typically decays over time. 
#
lrn_rate=0.00005 #0.0001
gamma=0.99

# Add these three lines instead:
epsilon_start  = 1.0
epsilon_end    = 0.05 #0.01
epsilon_decay  = 0.00001 #0.00003 #0.0001 #0.00005  # controls how fast it falls
gradient_clipping = 0.2 #0.5 #1.0

sequence_length = 2 #3
n_step_update = sequence_length - 1

finish_train = False

def save_info2cvs(csv_file:str, data:list, headers:list=None) -> None:
    """
    Docstring for save_info2cvs

    :param filename: Description
    :type filename: str
    :param data: Description
    :type data: list
    :param headers: Description
    :type headers: list
    """
    with open(csv_file, "w") as fd_write:
        if headers:
            fd_write.write(",".join(headers)+'\n')

        for ln in data:
            fd_write.write(",".join(["{:.3f}".format(l) for l in ln])+'\n')
        fd_write.close()

def create_layer(idx, lyr_size, lyr_bias, lyr_kernel, lyr_dropout) -> list:
    return [
        Dense(
            lyr_size,
            activation=tf.keras.activations.relu,
            name="LYR_{}".format(idx),
            kernel_initializer=lyr_kernel,
            bias_initializer=lyr_bias
            ),
        Dropout(lyr_dropout)
    ] if lyr_dropout > 0 else [
            Dense(
                lyr_size,
                activation=tf.keras.activations.relu,
                name="LYR_{}".format(idx),
                kernel_initializer=lyr_kernel,
                bias_initializer=lyr_bias
                )
    ]


def tensor_size(tnsr:any) -> any:
    #print("Size: {} {}".format(tnsr.shape.as_list(), len(tnsr.shape.as_list())))
    if len(tnsr.shape.as_list()) == 0:
        max_min = tnsr.maximum[()] - tnsr.minimum[()]
        return max_min+1
    elif len(tnsr.shape.as_list()) == 1:
        return tnsr.shape[0]

    return tnsr.shape[0]*tnsr.shape[1]

train_py_env = suite_gym.load(env_name)
train_py_env.reset()

eval_py_env = suite_gym.load(env_name)
eval_py_env.reset()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

num_actions = tensor_size(train_env.action_spec())
observations = tensor_size(train_env.time_step_spec().observation)

print("Action: {}".format(num_actions))
print("Observations: {}".format(observations))

print('Time Step Spec: {}'.format(train_env.action_spec()))
print('Time Step Spec: {}'.format(train_env.time_step_spec()))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
input_lr = Dense(observations, activation=None, name="Input")

layers = []
for idx in range(len(layer_sz)):
    layers = layers + create_layer(idx, layer_sz[idx], bias[idx], kernel_init[idx], dropout[idx])

"""
Output layer - number od units equal number of actions (4 in our case)
"""
q_values_layer = Dense(
    num_actions,
    activation= None, #tf.keras.activations.sigmoid,
    name="Output",
    kernel_initializer=kernel_init_lyr_out,
    bias_initializer=bias_lyr_out)

q_net = sequential.Sequential([input_lr] + layers + [q_values_layer], input_spec=train_env.time_step_spec().observation, name="QNet")

#
#Important parameters
# lerning rate - Adam optimizer parameter (0.001 - 0.0001)
# the discount factor (γ) of future rewards - gamma (0.9-1.0)
#
#Epsilon-Greedy Exploration: A strategy used during training to balance exploration
#(taking random actions to discover new possibilities) and exploitation (taking the action with the highest predicted Q-value).
#The probability of taking a random action, epsilon, typically decays over time. 

optimizer = tf.keras.optimizers.Adam(learning_rate=lrn_rate) #use lerning rate by default 0.001
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        gradient_clipping=gradient_clipping,
        gamma=gamma,
        epsilon_greedy=epsilon_start,
        n_step_update=n_step_update,
        td_errors_loss_fn=common.element_wise_huber_loss, #common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

agent.initialize()
agent.train = common.function(agent.train)

#print(q_net.trainable_variables)
total_gr = 0.0
grad = []
for vv in q_net.trainable_variables:
    n_par = np.linalg.norm(vv.numpy())
    grad.append(n_par)
    print("{} {:.4f} {:.4f}".format(vv.name, n_par**2, grad[-1]))
    total_gr += grad[-1]**2
total_gr=total_gr**0.5

print("Total Gr: {:.4f}".format(total_gr))
for p in grad:
    print("{:.2f}%".format((p**0.5/total_gr)*100))