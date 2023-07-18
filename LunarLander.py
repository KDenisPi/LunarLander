#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import mujoco_py as mj
import tensorflow as tf
from os.path import exists

"""
Reverb (dm-reverb) is an efficient and easy-to-use data storage and transport system designed for machine learning research.
Reverb is primarily used as an experience replay system for distributed reinforcement learning algorithms
but the system also supports multiple data structure representations such as FIFO, LIFO, and priority queues.
"""
import reverb

from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import tensor_spec
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.utils import common
from tf_agents.utils import composite
from tf_agents.policies import random_tf_policy
from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.metrics import tf_metrics

class LunarLander(object):

    def __init__(self, model_name=None) -> None:
        self.replay_buffer_max_length = 100000
        self.num_eval_episodes = 10
        self.num_iterations = 20000
        self.log_interval = 20 #200
        self.eval_interval = 1000
        self.batch_size = 64

        self.model_name = model_name

        self._env = gym.make('LunarLander-v2') #,
            # continuous: bool = False,
            # gravity: float = -10.0,
            # enable_wind: bool = False,
            # wind_power: float = 15.0,
            # turbulence_power: float = 1.5)

        self._py_env = suite_gym.wrap_env(self._env)
        self._tf_env = tf_py_environment.TFPyEnvironment(self._py_env)
        self.observations = self._tf_env.time_step_spec().observation.shape[0]
        self.ly_params = [self.observations, self.batch_size, self.batch_size] #[8, 64, 64]

        print('Time Step Spec: {}'.format(self._tf_env.time_step_spec()))
        print('Observation Spec: {}'.format(self._tf_env.time_step_spec().observation))
        print('Reward Spec: {}'.format(self._tf_env.time_step_spec().reward))
        print('Action Spec: {}'.format(self._tf_env.action_spec()))

    def save_model(self, model_name=None, q_net=None):
        """Save trained weights"""
        if model_name and q_net:
            q_net.save_weights(model_name)

    def load_model(self, model_name=None, q_net=None):
        """Load previosly saved weights"""
        if model_name and q_net and exists(model_name):
            q_net.load_weights(model_name)

    def prepare(self):
        """"""
        self.model_prepare()
        self.reply_buffer_prepare()

    def train_agent(self):
        """Train network"""
        self.agent.train_step_counter.assign(0)
        avg = self.compute_avg_return(self._tf_env, self.policy, self.num_eval_episodes)
        returns = [avg]
        #print('Average return {}'.format(avg))

        # Reset the environment.
        time_step = self._tf_env.reset()
        num_episodes = tf_metrics.NumberOfEpisodes()
        env_steps = tf_metrics.EnvironmentSteps()
        observers = [num_episodes, env_steps, self.rb_observer]

        driver = dynamic_episode_driver.DynamicEpisodeDriver(self._tf_env, self.policy, observers, num_episodes=2)

        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.batch_size,
            num_steps=2).prefetch(3)   #num_steps=2

        iterator = iter(self.dataset)

        for _ in range(self.num_iterations):

            # Collect a few steps and save to the replay buffer.
            time_step, _ = driver.run(time_step)
            experience, _ = next(iterator)

            batched_exp = tf.nest.map_structure(
                lambda t: composite.squeeze(t, axis=2),
                experience
            )

            train_loss = self.agent.train(batched_exp).loss
            step = self.agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % self.eval_interval == 0:
                avg_return = self.compute_avg_return(self._tf_env, self.agent.policy, self.num_eval_episodes)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
                returns.append(avg_return)

        #save trained weights
        self.save_model(self.model_name, self.q_net)

    def compute_avg_return(self, environment, policy, num_episodes=10):
        """"""
        total_return = 0.0
        for eps in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0
            steps = 0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
                steps += 1

            print('Episode: {} return: {} steps {}'.format(eps, episode_return, steps))
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def model_prepare(self):
        """Generate model
        Load previosly detected weights if needed
        """
        action_tensor_spec = self._tf_env.action_spec()
        time_step_tensor_spec = self._tf_env.time_step_spec()

        self.num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
        print('Number actions: {}'.format(self.num_actions))

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # its output.
        layers = [self.gen_layer(num_units) for num_units in self.ly_params]

        """
        Output layer - number od units equal number of actions (4 in our case)
        """
        q_values_layer = tf.keras.layers.Dense(
            self.num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))

        self.q_net = sequential.Sequential(layers + [q_values_layer])

        self.load_model(self.model_name, self.q_net)

        optimizer = tf.keras.optimizers.Adam() #use lerning rate by default 0.001
        self.train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DqnAgent(
            time_step_tensor_spec,
            action_tensor_spec,
            q_network=self.q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.train_step_counter)

        self.agent.initialize()

        self.gen_policy()

    def gen_layer(self, num_units : int) -> any:
        """
        VarianceScaling
        With distribution="truncated_normal" or "untruncated_normal",
        samples are drawn from a truncated/untruncated normal distribution with a mean of zero and
        a standard deviation (after truncation, if used) stddev = sqrt(scale / n), where n is:
        number of input units in the weight tensor, if mode="fan_in"
        """
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            )


    def gen_policy(self):
        """Generate polices"""
        self.policy = random_tf_policy.RandomTFPolicy(self._tf_env.time_step_spec(), self._tf_env.action_spec())

    def reply_buffer_prepare(self):
        """Prepre replay buffer"""
        replay_buffer_signature = tensor_spec.from_spec(self.agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

        table_name = 'uniform_table'
        table = reverb.Table(
            table_name,
            max_size=self.replay_buffer_max_length,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=None #replay_buffer_signature
            )

        self.reverb_server = reverb.Server([table])

        self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            data_spec=replay_buffer_signature,
            table_name=table_name,
            sequence_length=2,
            local_server=self.reverb_server)

        self.rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            self.replay_buffer.py_client,
            table_name,
            sequence_length=2)

if __name__ == '__main__':
    ll = LunarLander(model_name='lunarl_1')
    ll.prepare()
    ll.train_agent()
