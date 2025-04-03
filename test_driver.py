#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import base64

import numpy as np
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import tensor_spec
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.utils import common
from tf_agents.policies import random_py_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.drivers import py_driver
from tf_agents.metrics import tf_metrics
from tf_agents.networks.layer_utils import print_summary


env_name = "LunarLander-v2" # @param {type:"string"}
num_iterations = 250 # @param {type:"integer"}
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 2000 # @param {type:"integer"}

num_actions = 4

def collect_episode(environment, agent, num_episodes):
    """Collect data for episode"""
    driver = py_driver.PyDriver(
        environment,
        #py_tf_eager_policy.PyTFEagerPolicy(policy, use_tf_function=True),
        random_py_policy.RandomPyPolicy(environment.time_step_spec(), environment.action_spec()),
        [rb_observer],
        max_episodes=num_episodes)

    initial_time_step = environment.reset()
    driver.run(initial_time_step)


env = suite_gym.load(env_name)
env.reset()

#print('Observation Spec:')
#print(env.time_step_spec().observation)
#print('Action Spec:')
#print(env.action_spec())

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

observations = train_env.time_step_spec().observation.shape[0]

print('Time Step Spec: {}'.format(train_env.time_step_spec()))
print('Observation Num: {} Spec: {}'.format(observations, train_env.time_step_spec().observation))
print('Reward Spec: {}'.format(train_env.time_step_spec().reward))
print('Action Spec: {}'.format(train_env.action_spec()))


# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
input_lr = tf.keras.layers.Dense(observations, activation=None, name="Input")

nums_lyr_1 = observations
nums_lyr_2 = observations*2

layer_1 =  tf.keras.layers.Dense(
    nums_lyr_1,
    activation=tf.keras.activations.relu,
    name="LYR_1",
    kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=1.0 if nums_lyr_1 <= 10 else 2.0,
        mode='fan_in',
        distribution='truncated_normal')
    )

layer_2 =  tf.keras.layers.Dense(
    nums_lyr_2,
    activation=tf.keras.activations.relu,
    name="LYR_1",
    kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=1.0 if nums_lyr_2 <= 10 else 2.0,
        mode='fan_in',
        distribution='truncated_normal')
    )

"""
Output layer - number od units equal number of actions (4 in our case)
"""
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    name="Output",
    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))

q_net = sequential.Sequential([input_lr, layer_1, layer_2, q_values_layer])

#print("Q Net Input Spec: {}".format(self.q_net.input_tensor_spec))
#print("Q Net State Spec: {}".format(self.q_net.state_spec))

optimizer = tf.keras.optimizers.Adam() #use lerning rate by default 0.001
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

agent.initialize()
#self.agent.train = common.function(self.agent)
agent.train = common.function(agent.train)

"""Prepre replay buffer"""
replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

table_name = 'uniform_table'
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature
    )

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    data_spec=agent.collect_data_spec,
    table_name=table_name,
    sequence_length=None, #2
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2)

# Collect a few episodes using collect_policy and save to the replay buffer.
collect_episode(train_py_env, agent, collect_episodes_per_iteration)