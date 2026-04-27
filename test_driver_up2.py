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

env_name = 'LunarLander-v2' # @param {type:"string"}
#env_name='CartPole-v1'

num_iterations = 80000 if env_name == 'LunarLander-v2' else 25000
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

ckpt_restored=False
evaluate_chkpoint=None
for cmd in sys.argv:
    if cmd.find("ckpt") >= 0:
        evaluate_chkpoint = cmd
        break

run_idx = "up_16"
if len(sys.argv) >= 2:
    run_idx = sys.argv[1]

checkpoint_dir = './data/multi_checkpoint_{}'.format(run_idx)
results_file = './data/results_{}.dat'.format(run_idx)

result_actions_file = "./data/actions_{}.csv".format(run_idx)
result_info_file = "./data/actions_{}.csv".format(run_idx)

state_headers = ['Idx','X','Y','Vx','Vy','Angle','Va','LegL','LegR']
info_headers = ['Idx','Reward','Action','Last']

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


def collect_episode(environment, num_episodes=None, agent=None, num_steps=0, time_step=None) -> any:
    """Collect data for episode"""
    #print('Use policy: {}'.format("Agent" if agent else "Rendom"))

    collect_policy = py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True) if agent \
        else random_py_policy.RandomPyPolicy(environment.time_step_spec(), environment.action_spec())

    initial_time_step = time_step if time_step else environment.reset()
    #print("First step: first {} last {}".format(initial_time_step.is_first(), initial_time_step.is_last()))

    driver = py_driver.PyDriver(
        env=environment,
        policy=collect_policy,
        observers=[rb_observer],
        end_episode_on_boundary=True,
        max_steps=num_steps,
        max_episodes=num_episodes)

    last_time_step, policy_state = driver.run(initial_time_step)
    return last_time_step
    #print("Last step: {} Policy: {}".format(last_time_step, policy_state))


def compute_avg_return(environment, policy, num_episodes=10):
    #print("Started compute_avg_return")
    #print(tf.keras.backend.learning_phase())

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

def save_parameters(StTime, NumIter, BatchSize, UpTau, UpPrd, LrnRate, Gamma, Epsilon, GradClip, SeqLen, RefillIntr, InitRecs, Layrs, Bias, DrpOut, results) -> None:
    filename = "./data/parameters.csv"
    headers=['Date', 'Duration','NumIterations', 'BatchSize','UpTau', 'UpPrd', 'LrnRate', 'Gamma', 'Epsilon', 'GradClip', 'RefillIntr', 'InitRecords', 'Layrs']

    if os.path.exists(filename):
        headers = None

    with open(filename, 'a') as file:
        if headers:
            file.write(",".join(headers)+'\n')
        file.write("{},{},{},{},{},{},{},{},{:0.2f},{},{},{},{},{},".format(datetime.now(), (datetime.now() - StTime), NumIter, BatchSize,
                        UpTau, UpPrd, LrnRate, Gamma, Epsilon, GradClip, SeqLen, RefillIntr, InitRecs, len(Layrs)))
        file.write(",".join(["{}".format(lr) for lr in Layrs])+',')
        file.write(",".join(["{:0.2f}".format(bs.get_config()['value']) for bs in Bias])+',')
        file.write(",".join(["{:0.2f}".format(drpout) for drpout in DrpOut])+',')
        file.write(",".join(["{:0.2f}".format(res) for res in results])+'\n')

def param_names(q_net) -> list:
    val = ['Step']
    for vv in q_net.trainable_variables:
        val.append(vv.name)
    val.append('Total')
    return val

def param_gradients(step, q_net, grads:list) -> None:
    val = []
    total_gr = 0.0

    val.append(step)
    for vv in q_net.trainable_variables:
        val.append(np.linalg.norm(vv.numpy()))
        total_gr += val[-1]**2
    total_gr=total_gr**0.5
    val.append(total_gr)
    grads.append(val)

#Set CTRL+C handler
signal.signal(signal.SIGINT, handler)

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

"""
weight_decay	Float. If set, weight decay is applied.
clipnorm	Float. If set, the gradient of each weight is individually clipped so that its norm is no higher than this value.
clipvalue	Float. If set, the gradient of each weight is clipped to be no higher than this value.
global_clipnorm	Float. If set, the gradient of all weights is clipped so that their global norm is no higher than this value.
"""

optimizer = tf.keras.optimizers.Adam(learning_rate=lrn_rate, clipnorm=gradient_clipping)
train_step_counter = tf.Variable(0)

"""
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        gradient_clipping=gradient_clipping,

target_update_tau	Factor for soft update of the target networks.
target_update_period	Period for soft update of the target networks.

gradient_clipping	Norm length to clip gradients.
gradient_clipping: Optional[types.Float] = None,

target_update_tau=0.001 #0.005 #0.05	#Factor for soft update of the target networks.
target_update_period=30 #10 #5 	        #Period for soft update of the target networks.
gradient_clipping = 0.5

tf.keras.optimizers.Adam
weight_decay	Float. If set, weight decay is applied.
clipnorm	Float. If set, the gradient of each weight is individually clipped so that its norm is no higher than this value.
clipvalue	Float. If set, the gradient of each weight is clipped to be no higher than this value.
global_clipnorm	Float. If set, the gradient of all weights is clipped so that their global norm is no higher than this value.

Suggest:
gradient_clipping=None #gradient_clipping,
optimizer = tf.keras.optimizers.Adam(learning_rate=lrn_rate,
    clipnorm=gradient_clipping)

"""

#Step,
# QNet/Input/kernel:0,
# QNet/Input/bias:0,
# QNet/LYR_0/kernel:0,
# QNet/LYR_0/bias:0,
# QNet/LYR_1/kernel:0,
# QNet/LYR_1/bias:0,
# QNet/Output/kernel:0,
# QNet/Output/bias:0,
# Total

# Layers you want clipped — match by name prefix
CLIP_LAYER_NAMES = ["LYR_"]   # hidden layers only; excludes "Input" and "Output"
CLIP_NORM_VALUE  = gradient_clipping   # e.g. 0.2

class SelectiveClipDqnAgent(dqn_agent.DqnAgent):
    """DQN agent that applies clipnorm only to selected layers."""

    def _train(self, experience, weights=None):
        """Override to apply per-layer gradient clipping."""
        with tf.GradientTape() as tape:
            loss_info = self._loss(experience, weights=weights, training=True)

            variables = self._q_network.trainable_variables
            gradients = tape.gradient(loss_info.loss, variables)

            # Clip only the layers whose name starts with a target prefix
            clipped_gradients = []
            for grad, var in zip(gradients, variables):
                if grad is None:
                    clipped_gradients.append(grad)
                elif any(lyr in var.name for lyr in CLIP_LAYER_NAMES):
                    clipped_gradients.append(tf.clip_by_norm(grad, CLIP_NORM_VALUE))
                else:
                    clipped_gradients.append(grad)   # unclipped

            self._optimizer.apply_gradients(zip(clipped_gradients, variables))
            self.train_step_counter.assign_add(1)
            return loss_info


agent = SelectiveClipDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        gradient_clipping=None, #gradient_clipping,
        gamma=gamma,
        epsilon_greedy=epsilon_start,
        n_step_update=n_step_update,
        td_errors_loss_fn=common.element_wise_huber_loss, #common.element_wise_squared_loss,
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
    sampler=reverb.selectors.Uniform(), #reverb.selectors.Lifo(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature
    )

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    data_spec=agent.collect_data_spec,
    table_name=table_name,
    sequence_length=sequence_length,
    dataset_buffer_size=batch_size*sequence_length,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=sequence_length)


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

    if evaluate_chkpoint:
        evt_ckpnt = "{}/{}".format(checkpoint_dir,evaluate_chkpoint)
        print(evt_ckpnt)
        ckpt_mng_last = evt_ckpnt if evt_ckpnt in ckpt_manager.checkpoints else ckpt_manager.latest_checkpoint

    print("Available checkpoints: {}".format(ckpt_manager.checkpoints))
    if ckpt_mng_last is not None:
        print("Restore Ckpt from: {}".format(ckpt_mng_last))
        ckpt.restore(ckpt_mng_last).expect_partial()
        ckpt_restored=True
    else:
        print("No checkpoints")

    print("Loaded checkpoint from: {} Step: {} Save counter: {}".format(checkpoint_dir, train_step_counter.numpy(), ckpt.save_counter.numpy()))

stat_actions = []
stat_info = []

stat_cntr = 0

if evaluate_chkpoint:
    if not ckpt_restored:
        print("Model was not trainted, nothing to evaluate")
        exit()

    eval_result = []
    for _ in range(3):
        eval_result.append(compute_avg_return(eval_env, agent.policy, num_eval_episodes))
    print(eval_result)
    exit()

print("Start training.....")
print_summary(q_net)

train_collect_policy = py_tf_eager_policy.PyTFEagerPolicy(agent.collect_policy, use_tf_function=True)
#train_collect_policy = random_py_policy.RandomPyPolicy(train_py_env.time_step_spec(), train_py_env.action_spec())

train_driver = py_driver.PyDriver(
    env=train_py_env,
    policy=train_collect_policy,
    observers=[rb_observer],
    end_episode_on_boundary=True,
    max_steps=train_driver_max_step,
    max_episodes=0)

policy_state = train_collect_policy.get_initial_state(train_py_env.batch_size)

#put initial number of records to buffer
train_time_step = collect_episode(train_py_env, num_steps=num_initial_records)

f_step = agent.train_step_counter.numpy()
print("Frames in reply buffer: {} First step: {}".format(replay_buffer.num_frames(), f_step))

returns = read_results(results_file)
print(returns)

avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns.append(avg_return)

tm_start = datetime.now()

reward_counter = 0.0
loss_counter = 0.0

loss_list = []
grads = []

episodes_trj = 0
boundary_trj = 0

param_gradients(0,q_net,grads)

iterator = iter(replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=sequence_length))

for _ in range(num_iterations):
    # Collect a few episodes using collect_policy and save to the replay buffer.
    #changed num_steps = batch_size to 0 Use to episodes = 1 instead 0
    #modified - no agent - random

    train_time_step, policy_state = train_driver.run(
        time_step=train_time_step,
        policy_state=policy_state,
    )

    num_frames = replay_buffer.num_frames()

    # Use data from the buffer and update the agent's network.
    trajectories, _ = next(iterator)
    train_loss = agent.train(experience=trajectories)

    reward_per_batch = (np.sum(trajectories.reward.numpy())/batch_size)
    reward_counter = reward_counter + reward_per_batch

    loss_counter = loss_counter + train_loss.loss

    step = agent.train_step_counter.numpy()

    # Decay epsilon each step
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-epsilon_decay * step)
    agent.collect_policy._epsilon = epsilon  # inject updated value

    #loss_list.append([step, train_loss.loss, reward_per_batch])
    if step > 0 and step % log_loss_interval == 0:
        loss_list.append([step, train_loss.loss])
        param_gradients(step,q_net,grads)


    if step % log_interval == 0:
        print('step = {0}: loss = {1:0.3f} Reward: {2:0.3f} ε={3:.4f} Sec. {4} Frames: {5}'.format(step, train_loss.loss, reward_per_batch, epsilon, (datetime.now()-tm_start).seconds, num_frames))
        #if step > 0:
        #    print('step = {0}: Avg.Loss = {1:0.3f} Avg.Reward: {2:0.3f} Sec. {3}'.format(
        #        step, loss_counter/(step-f_step), reward_counter/(step-f_step), (datetime.now()-tm_start).seconds))

    if flush_interval > 0 and step % flush_interval == 0:
        rb_observer.flush()

    if step > 0 and step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
        returns.append(avg_return)
        print('---> Step = {0}: Average Return = {1:0.2f} All: {2}'.format(step, avg_return, returns))

    if step > 0 and step % episode_for_checkpoint == 0:
        if ckpt:
            ckpt.step.assign_add(1)
            sv_folder = ckpt_manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), sv_folder))

    if finish_train:
        break

avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns.append(avg_return)
print('---> Step = {0}: Average Return = {1:0.2f} All: {2}'.format(step, avg_return, returns))

if ckpt:
    ckpt.step.assign_add(1)
    sv_folder = ckpt_manager.save()
    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), sv_folder))
    #ckpt.save(global_step=train_step_counter)


save_results(results_file, returns)
save_info2cvs("./data/loss.csv", loss_list, ["Step", "Loss"])

prm_headrs = param_names(q_net)
save_info2cvs("./data/pgradients.csv", grads, prm_headrs)

save_parameters(tm_start, num_iterations, batch_size, target_update_tau, target_update_period, lrn_rate, gamma, epsilon,
                gradient_clipping, sequence_length, refill_buffer_interval, num_initial_records, layer_sz, bias, dropout, returns)

print("Training finished..... {}".format(datetime.now() - tm_start))
print(returns)
