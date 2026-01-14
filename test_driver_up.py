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
num_iterations = 1000
collect_episodes_per_iteration = 2 # @param {type:"integer"}
replay_buffer_capacity = 150000 # @param {type:"integer"}

batch_size = 64

num_eval_episodes = 10 # @param {type:"integer"}
eval_interval = 2000 # 100 @param {type:"integer"}
log_interval = 1000 # 50 @param {type:"integer"}
log_episode_interval = 5

layer_sz = [256, 128]
bias = [tf.keras.initializers.Constant(-0.2), tf.keras.initializers.Constant(-0.2)]  #tf.keras.initializers.Constant(-0.2)
lrn_rate=0.0001
gamma=0.99
epsilon=0.95 #0.995


trace = False
trace_fld = '/home/denis/sources/LunarLander/logs' if trace else ''

#checkpoints
ckpt_max_to_keep = 5
episode_for_checkpoint = 250
checkpoint_dir = './data/multi_checkpoint_up_4'

results_file = './data/results_up_4.dat'

finish_train = False

def handler(signum, frame):
    """Signal processing handler"""
    signame = signal.Signals(signum).name
    print(f'Signal handler called with signal {signame} ({signum})')
    global finish_train
    finish_train = True

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
        file.close()
    return results


def collect_episode(environment, num_episodes, agent, num_steps=0):
    """Collect data for episode"""
    #print('Use policy: {}'.format("Agent" if agent else "Rendom"))

    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True) if agent \
                else random_py_policy.RandomPyPolicy(environment.time_step_spec(),
                    environment.action_spec())

    initial_time_step = environment.reset()
    #print("First step: first {} last {}".format(initial_time_step.is_first(), initial_time_step.is_last()))

    driver = py_driver.PyDriver(
        env=environment,
        policy=collect_policy,
        observers=[rb_observer],
        end_episode_on_boundary=True,
        max_steps=num_steps,
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

        print('Evaluation episode: {0} Rewards: {1:0.2f} {2} steps {3} Duration {4} sec'.format(
            eps,
            episode_return.numpy()[0],
            time_step.reward.numpy(),
            steps,
            (datetime.now()-tm_start).seconds)
            )

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
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2)


ckpt = None
ckpt_manager = None

if checkpoint_dir:
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1),
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=train_step_counter
    )
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=ckpt_max_to_keep)
    ckpt_mng_last = ckpt_manager.latest_checkpoint
    if ckpt_mng_last is not None:
        print("Restore Ckpt from: {}".format(ckpt_mng_last))
        ckpt.restore(ckpt_mng_last)
    else:
        print("No checkpoints")

    #ckpt.initialize_or_restore()
    print("Loaded checkpoint from: {} Step: {} Save counter: {}".format(checkpoint_dir, train_step_counter.numpy(), ckpt.save_counter.numpy()))
    #exit()


print("Start training.....")
print_summary(q_net)

returns = read_results(results_file)
print(returns)

avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns.append(avg_return)

tm_start = datetime.now()

replay_buffer.clear()

f_step = agent.train_step_counter.numpy()
print("Frames in reply buffer: {} First step: {}".format(replay_buffer.num_frames(), f_step))

iterator = iter(replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=2))


rb_observer.flush()
tm_start = datetime.now()

reward_counter = 0.0
loss_counter = 0.0

episodes_trj = 0
boundary_trj = 0

for _ in range(num_iterations):
    # Collect a few episodes using collect_policy and save to the replay buffer.
    collect_episode(train_py_env, num_episodes=0, agent=agent, num_steps=batch_size)#collect_episodes_per_iteration, agent)

    num_frames = replay_buffer.num_frames()
    #print("Step: {} Frames in reply buffer: {}".format(step, num_frames))

    # Use data from the buffer and update the agent's network.
    trajectories, _ = next(iterator)
    train_loss = agent.train(experience=trajectories)

    reward_counter = reward_counter + np.sum(trajectories.reward.numpy())
    loss_counter = loss_counter + train_loss.loss

    step = agent.train_step_counter.numpy()
    if step % log_interval == 0:
        print('step = {0}: loss = {1:0.2f} Reward: {2:0.2f}'.format(step, train_loss.loss, np.sum(trajectories.reward.numpy())))
        if step > 0:
            print('step = {0}: Avg.Loss = {1:0.2f} Avg.Reward: {2:0.2f} Sec. {3}'.format(
                step, loss_counter/(step-f_step), reward_counter/(step-f_step), (datetime.now()-tm_start).seconds))

    if step % 200 == 0:
        rb_observer.flush()

    if step % 100 == 0:
        print("Step: {} Frames in reply buffer: {}".format(step, num_frames))


    if step > 0 and step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        returns.append(avg_return)
        print('---> Step = {0}: Average Return = {1:0.2f} All: {2}'.format(step, avg_return, returns))

    if step % episode_for_checkpoint == 0:
        if ckpt:
            ckpt.step.assign_add(1)
            sv_folder = ckpt_manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), sv_folder))


    if finish_train:
        break

if ckpt:
    ckpt.step.assign_add(1)
    sv_folder = ckpt_manager.save()
    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), sv_folder))
    #ckpt.save(global_step=train_step_counter)


save_results(results_file, returns)

print("Training finished..... {}".format(datetime.now() - tm_start))
print(returns)
#print_summary(q_net)
