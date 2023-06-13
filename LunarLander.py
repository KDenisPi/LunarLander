#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import numpy as np
import mujoco_py as mj
import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.specs import tensor_spec
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.specs import tensor_spec

class LunarLander(object):

    def __init__(self) -> None:
        self._env = gym.make('LunarLander-v2') #,
            # continuous: bool = False,
            # gravity: float = -10.0,
            # enable_wind: bool = False,
            # wind_power: float = 15.0,
            # turbulence_power: float = 1.5)

        self._py_env = suite_gym.wrap_env(self._env)
        #self._py_env = suite_gym.load('LunarLander-v2')
        self._tf_env = tf_py_environment.TFPyEnvironment(self._py_env)
        self.ly_params = [64, 128]

        print(self._tf_env.time_step_spec())

        print('Observation Spec:')
        print(self._tf_env.time_step_spec().observation)

        print('Reward Spec:')
        print(self._tf_env.time_step_spec().reward)

        print('Action Spec:')
        print(self._tf_env.action_spec())

    def run(self):
        """"""
        time_step = self._tf_env.reset()

        print('Time step:')
        print(time_step)

        action_step = self.policy.action(time_step)
        print('Action step:')
        print(action_step)

        next_time_step = self._tf_env.step(action_step)
        print('Next time step:')
        print(next_time_step)

    def model_prepare(self):
        action_tensor_spec = self._tf_env.action_spec()
        time_step_tensor_spec = self._tf_env.time_step_spec()

        print("time_step_tensor_spec")
        print(time_step_tensor_spec)
        print("action_tensor_spec")
        print(action_tensor_spec)

        self.num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
        print("num_actions")
        print(self.num_actions)

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
        optimizer = tf.keras.optimizers.Adam() #use lerning rate by default 0.001
        self.train_step_counter = tf.Variable(0)


        #action_spec = tensor_spec.from_spec(self._tf_env.action_spec())

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
        #action_tensor_spec = tensor_spec.from_spec(self._tf_env.action_spec())
        #ime_step_tensor_spec = tensor_spec.from_spec(self._tf_env.time_step_spec())

        self.policy = random_tf_policy.RandomTFPolicy(self._tf_env.time_step_spec(), self._tf_env.action_spec())



if __name__ == '__main__':
    ll = LunarLander()
    ll.model_prepare()
    ll.run()


