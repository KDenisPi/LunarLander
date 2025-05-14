#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import base64
from datetime import datetime

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
from tf_agents.networks.layer_utils import print_summary


env_name = "LunarLander-v2" # @param {type:"string"}
num_iterations = 2000 # @param {type:"integer"}
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 2000 # @param {type:"integer"}

num_actions = 4
num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 100 # @param {type:"integer"}
log_interval = 50 # @param {type:"integer"}

def collect_episode(environment, num_episodes, agent = None):
    """Collect data for episode"""
    #print('Use policy: {}'.format("Agent" if agent else "Rendom"))
    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True) if agent else random_py_policy.RandomPyPolicy(environment.time_step_spec(), environment.action_spec())

    driver = py_driver.PyDriver(
        environment,
        collect_policy,
        [rb_observer],
        max_episodes=num_episodes)

    initial_time_step = environment.reset()
    driver.run(initial_time_step)


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    print("Started compute_avg_return")

    for eps in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        steps = 0
        tm_start = datetime.now()

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            steps = steps + 1
        total_return += episode_return
        print('Episode: {0} return: {1:0.2f} steps {2} Duration {3} sec'.format(eps, episode_return.numpy()[0], steps, (datetime.now()-tm_start).seconds))

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


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

fc_layer_params = 8 #sum of all parts of trajectory

observations = train_env.time_step_spec().observation.shape[0]

print('Time Step Spec: {}'.format(train_env.time_step_spec()))
print('Observation Num: {} Spec: {}'.format(observations, train_env.time_step_spec().observation))
print('Reward Spec: {}'.format(train_env.time_step_spec().reward))
print('Action Spec: {}'.format(train_env.action_spec()))


# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
input_lr = tf.keras.layers.Dense(fc_layer_params, activation=None, name="Input")

nums_lyr_1 = fc_layer_params*10
nums_lyr_2 = fc_layer_params*20

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
    name="LYR_2",
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

#print_summary(q_net)

#print("Q Net Input Spec: {}".format(self.q_net.input_tensor_spec))
#print("Q Net State Spec: {}".format(self.q_net.state_spec))

optimizer = tf.keras.optimizers.Adam() #use lerning rate by default 0.001
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=None, #common.element_wise_squared_loss,
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


avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

print("Start training.....")
print_summary(q_net)

#for l_n in range(0,4):
#    lyr = q_net.get_layer(index=l_n)
#    print('Name: {} Weights: {}'.format(lyr.name, lyr.get_weights()))

#exit()

tm_start = datetime.now()
for nm_it in range(num_iterations):

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(train_py_env, collect_episodes_per_iteration, agent)

    # Use data from the buffer and update the agent's network.
    iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
    trajectories, _ = next(iterator)

    #print('Trajectories {}'.format(trajectories))
    #break
    train_loss = agent.train(experience=trajectories)

    replay_buffer.clear()

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
        print('step = {0}: loss = {1} Duration {2} sec'.format(step, train_loss.loss, (datetime.now()-tm_start).seconds))
        tm_start = datetime.now()

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)
