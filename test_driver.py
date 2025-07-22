#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import base64
from datetime import datetime
import signal

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
episodes_for_training = 4000
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 2000000 # @param {type:"integer"}

num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 150000 # 100 @param {type:"integer"}
log_interval = 50000 # 50 @param {type:"integer"}

layer_sz = [256, 128]
bias = [None, None]  #tf.keras.initializers.Constant(-0.2)
lrn_rate=0.0001
gamma=0.99
epsilon=0.1


trace = False
trace_fld = '/home/denis/sources/LunarLander/logs' if trace else ''

finish_train = False

def handler(signum, frame):
    """Signal processing handler"""
    signame = signal.Signals(signum).name
    print(f'Signal handler called with signal {signame} ({signum})')
    global finish_train
    finish_train = True


def collect_episode(environment, num_episodes, agent):
    """Collect data for episode"""
    #print('Use policy: {}'.format("Agent" if agent else "Rendom"))

    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True) if agent else random_py_policy.RandomPyPolicy(environment.time_step_spec(),
                                                                                                  environment.action_spec())

    initial_time_step = environment.reset()
    #print("First step: first {} last {}".format(initial_time_step.is_first(), initial_time_step.is_last()))

    driver = py_driver.PyDriver(
        env=environment,
        policy=collect_policy,
        observers=[rb_observer],
        end_episode_on_boundary=True,
        #max_steps=0,
        max_episodes=num_episodes)

    last_time_step, policy_state = driver.run(initial_time_step)
    #print("Last step: first {} last {}".format(last_time_step.is_first(), last_time_step.is_last()))


def compute_avg_return(environment, policy, num_episodes=10):
    #print("Started compute_avg_return")

    total_return = 0.0
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
        """
        print('Episode: {0} Rewards: {1:0.2f} {2} steps {3} Duration {4} sec'.format(
            eps,
            episode_return.numpy()[0],
            time_step.reward.numpy(),
            steps,
            (datetime.now()-tm_start).seconds)
            )
        """

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


#Set CTRL+C handler
signal.signal(signal.SIGINT, handler)

train_py_env = suite_gym.load(env_name)
train_py_env.reset()

eval_py_env = suite_gym.load(env_name)
eval_py_env.reset()

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

action_tensor_spec = train_env.action_spec()
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

observations = train_env.time_step_spec().observation.shape[0]

print("Num actions: {}".format(num_actions))
print("Num observations: {}".format(observations))

print('Time Step Spec: {}'.format(train_env.time_step_spec()))
print('Observation Spec: {}'.format(train_env.time_step_spec().observation))
print('Reward Spec: {}'.format(train_env.time_step_spec().reward))
print('Action Spec: {}'.format(train_env.action_spec()))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
input_lr = tf.keras.layers.Dense(observations, activation=None, name="Input")

layer_1 =  tf.keras.layers.Dense(
    layer_sz[0],
    activation=tf.keras.activations.relu,
    name="LYR_1",
    kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=1.0,
        mode='fan_in',
        distribution='truncated_normal'),
    bias_initializer=bias[0]
    )

layer_2 =  tf.keras.layers.Dense(
    layer_sz[1],
    activation=tf.keras.activations.relu,
    name="LYR_2",
    kernel_initializer=tf.keras.initializers.VarianceScaling(
        scale=1.0,
        mode='fan_in',
        distribution='truncated_normal'),
    bias_initializer=bias[1]
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

q_net = sequential.Sequential([input_lr, layer_1, layer_2, q_values_layer], input_spec=train_env.time_step_spec().observation, name="QNet")

#
#Important parameters
# lerning rate - Adam optimizer parameter (0.001 - 0.0001)
# the discount factor (γ) of future rewards - gamma (0.9-1.0)
#
optimizer = tf.keras.optimizers.Adam(learning_rate=lrn_rate) #use lerning rate by default 0.001
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        gamma=gamma,
        epsilon_greedy=epsilon,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

agent.initialize()
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


print("Start training.....")
print_summary(q_net)

#for l_n in range(0,4):
#    lyr = q_net.get_layer(index=l_n)
#    print('Name: {} Trainamle: {} Config: {}'.format(lyr.name, lyr.trainable, lyr.get_config()))
#    #print('Name: {} Weights: {}'.format(lyr.name, lyr.get_weights()))

#exit()

avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

tm_g_start = datetime.now()
tm_start = datetime.now()

replay_buffer.clear()

step = agent.train_step_counter.numpy()
print("Frames in reply buffer: {} Step: {}".format(replay_buffer.num_frames(), step))

loss_overflow = False

if trace:
    pr_options = tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=3,
        python_tracer_level=1,
        device_tracer_level=1,
        delay_ms=None
    )

    tf.profiler.experimental.start(trace_fld, options=pr_options)


iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))  #sample_batch_size - TODO

episode = 0
while episode < episodes_for_training:

    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(train_py_env, collect_episodes_per_iteration, agent)
    num_frames = replay_buffer.num_frames()
    #print("Episode: {} Frames in reply buffer: {}".format(episode, num_frames))

    episode = episode + 1

    if num_frames == 0:
        break


    """
    iterator = iter(replay_buffer.as_dataset(sample_batch_size=1))
    print("Iterator element spec: {}".format(iterator.element_spec))

    while counter <= num_frames:
        trajectories, _ = next(iterator)
        #print(trajectories)
        print("{0} Action:{1} Reward:{2} Step type: {3}".format(counter,
                                                            trajectories.action.numpy(),
                                                            trajectories.reward.numpy(),
                                                            trajectories.step_type.numpy()))
        counter = counter + 1
        #break

    #replay_buffer.clear()
    #print("Frames in reply buffer: {}".format(replay_buffer.num_frames()))
    exit()
    """

    reward_counter = 0.0
    loss_counter = 0.0

    episodes_trj = 0
    boundary_trj = 0

    counter = 0
    while step < num_frames:
        # Use data from the buffer and update the agent's network.
        trajectories, _ = next(iterator)

        counter = counter + 1
        """
        print("Ac:{0} Rw:{1} Stp:{2} NStp: {3} Dsc:{4} Last: {5} Boundary: {6} {7}".format(
                                                        trajectories.action.numpy(),
                                                        trajectories.reward.numpy(),
                                                        trajectories.step_type.numpy(),
                                                        trajectories.next_step_type.numpy(),
                                                        trajectories.discount.numpy(),
                                                        trajectories.is_last().numpy(),
                                                        trajectories.is_boundary().numpy(),
                                                        np.sum(trajectories.is_boundary())))
        """
        #print(trajectories)

        if np.sum(trajectories.is_last()):
            episodes_trj = episodes_trj + 1
            #print("----> End of Episode step: {}".format(counter))
        if np.sum(trajectories.is_boundary()):
            boundary_trj = boundary_trj + 1
            #print("----> End of Boundary step: {}".format(counter))
        #continue

        step = agent.train_step_counter.numpy()

        if trace:
            with tf.profiler.experimental.Trace("Train", step_num=step):
                train_loss = agent.train(experience=trajectories)
        else:
            train_loss = agent.train(experience=trajectories)


        reward_counter = reward_counter + np.sum(trajectories.reward.numpy())
        loss_counter = loss_counter + train_loss.loss
        #print("Episode {0} Step: {1} Loss: {2:0.2f} Reward: {3:0.2f}".format(episode, step, train_loss.loss, np.sum(trajectories.reward.numpy())))


        if np.sum(trajectories.reward.numpy()) > 10000:
            loss_overflow = True

        if loss_overflow:
            print("Ac:{0} Rw:{1} Stp:{2} NStp: {3} Dsc:{4} Last: {5} Boundary: {6} {7}".format(
                                                            trajectories.action.numpy(),
                                                            trajectories.reward.numpy(),
                                                            trajectories.step_type.numpy(),
                                                            trajectories.next_step_type.numpy(),
                                                            trajectories.discount.numpy(),
                                                            trajectories.is_last().numpy(),
                                                            trajectories.is_boundary().numpy(),
                                                            np.sum(trajectories.is_boundary())))



        if step % log_interval == 0:
            print('step = {0}: loss = {1:0.2f} Reward: {2:0.2f} Duration {3} sec Episode: {4}'.format(
                step, train_loss.loss, np.sum(trajectories.reward.numpy()), (datetime.now()-tm_start).seconds, episode))
            tm_start = datetime.now()


        #if np.sum(trajectories.is_boundary()):
        #    print("Break by boundary. Episode: {0} Current step: {1} Frames: in reply buffer: {2} Last:{3} Bnd:{4} Counter: {5}".format(
        #        episode, step, num_frames, episodes_trj, boundary_trj, counter))
        #    break

        #if np.sum(trajectories.is_last()):
        #    print("Break by last. Episode: {0} Current step: {1} Frames: in reply buffer: {2} Last:{3} Bnd:{4} Counter: {5}".format(
        #        episode, step, num_frames, episodes_trj, boundary_trj, counter))
        #    break


        #if counter > num_frames:
        #    print("Break by last frame. Episode: {0} Current step: {1} Frames: in reply buffer: {2} Last:{3} Bnd:{4} Counter:{5}".format(
        #        episode, step, num_frames, episodes_trj, boundary_trj, counter))
        #    break

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            returns.append(avg_return)
            print('step = {0}: Average Return = {1:0.2f} All: {2}'.format(step, avg_return, returns))

        if finish_train:
            break

    if loss_overflow:
        break

    #print("Episodes: {} Boundary: {}".format(episodes_trj, boundary_trj))
    #exit()

    #print("Episode: {0} Current step: {1} Frames in reply buffer: {2} Reward: {3:0.2f} Loss: {4:0.2f}".format(
    # episode, step, num_frames, reward_counter/num_frames, loss_counter/num_frames))

    print("Episode: {0} Current step: {1} Frames in reply buffer: {2} Counter: {3} Reward: {4:0.2f} Loss: {5:0.2f} {6} {7}".format(
        episode, step, num_frames, counter, reward_counter/counter, loss_counter/counter, episodes_trj, boundary_trj))

    if finish_train:
        break

if trace:
    tf.profiler.experimental.stop()

print("Training finished..... {}".format(datetime.now() - tm_g_start))
print(returns)
#print_summary(q_net)
