#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional
from xmlrpc.client import Boolean
import gym
import numpy as np
#import mujoco_py as mj
import tensorflow as tf
import os
import sys
import signal
from datetime import datetime

"""
Reverb (dm-reverb) is an efficient and easy-to-use data storage and transport system designed for machine learning research.
Reverb is primarily used as an experience replay system for distributed reinforcement learning algorithms
but the system also supports multiple data structure representations such as FIFO, LIFO, and priority queues.
"""
import reverb

from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.specs import tensor_spec
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import sequential
from tf_agents.utils import common
from tf_agents.utils import composite
from tf_agents.policies import random_tf_policy
from tf_agents.policies import PolicySaver
from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.metrics import tf_metrics
from tf_agents.networks.layer_utils import print_summary

class LunarLander(object):
    """Reinforsment learning for Lunar Lander environment"""

    #Correctly finish train by CTRL+C
    bFinishTrain = False

    def handler(signum, frame):
        """Signal processing handler"""
        signame = signal.Signals(signum).name
        print(f'Signal handler called with signal {signame} ({signum})')
        LunarLander.bFinishTrain = True

    def dt() -> str:
        return str(datetime.now())

    def __init__(self, model_name: Optional [str] = None,
                 num_iterations: Optional[int] = 20000,
                 replay_buffer_max_length: Optional[int] = 100000,
                 eval_interval: Optional [int] = 100) -> None:
        """Initialization"""
        self.replay_buffer_max_length = replay_buffer_max_length
        self.num_eval_episodes = 10
        self.num_iterations = num_iterations
        self.log_interval = 20 #200
        self.eval_interval = eval_interval #1000
        self.batch_size = 128

        self.debug = True
        self.debug_data = 'data/'

        self.model_name = os.path.join(".", model_name) if model_name else None
        self.checkpoint_dir = self.model_name + '.checkpoint' if self.model_name else None
        self.policy_dir = self.model_name + '.policy' if self.model_name else None

        self._env = gym.make('LunarLander-v2') #,
            # continuous: bool = False,
            # gravity: float = -10.0,
            # enable_wind: bool = False,
            # wind_power: float = 15.0,
            # turbulence_power: float = 1.5)
        self._env_eval = gym.make('LunarLander-v2')

        self._py_env = suite_gym.wrap_env(self._env)
        self._tf_env = tf_py_environment.TFPyEnvironment(self._py_env)

        self._py_env_eval = suite_gym.wrap_env(self._env_eval)
        self._tf_env_eval = tf_py_environment.TFPyEnvironment(self._py_env_eval)


        self.observations = self._tf_env.time_step_spec().observation.shape[0]
        self.ly_params = [[self.observations*4, 0.2], [self.observations*2, 0.0]]

        if self.is_debug:
            print('Time Step Spec: {}'.format(self._tf_env.time_step_spec()))
            print('Observation Num: {} Spec: {}'.format(self.observations, self._tf_env.time_step_spec().observation))
            print('Reward Spec: {}'.format(self._tf_env.time_step_spec().reward))
            print('Action Spec: {}'.format(self._tf_env.action_spec()))

    @property
    def mname(self) -> any:
        return self.model_name

    @property
    def is_debug(self) -> bool:
        return self.debug

    def save_model(self, agent):
        """Save trained weights"""
        if not self.mname:
            return

        print("Save checkpoint to: {} Step: {}".format(self.checkpoint_dir, self.train_step_counter))
        self.train_checkpointer.save(global_step=self.train_step_counter)

    def load_model(self, agent) -> None:
        if not self.mname:
            return
        """Load previosly saved weights"""
        print("Loading checkpoint from: {} Step: {}".format(self.checkpoint_dir, self.train_step_counter))
        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.train_step_counter
        )
        self.train_checkpointer.initialize_or_restore()
        print("Loaded checkpoint from: {} Step: {}".format(self.checkpoint_dir, self.train_step_counter))


    def prepare(self):
        """"""
        self.model_prepare()
        self.reply_buffer_prepare()
        self.load_model(self.agent)


    def train_agent(self) -> any:
        """Train network"""
        if self.is_debug:
            print("Counters before. Agent: {} Saved: {}".format(self.agent.train_step_counter, self.train_step_counter))
        self.agent.train_step_counter.assign(self.train_step_counter)

        if self.is_debug:
            print_summary(self.q_net)

        avg = self.compute_avg_return(self._tf_env_eval, self.agent.policy, self.num_eval_episodes)

        if self.is_debug:
            print('step = {0}: Average Return = {1:0.2f}'.format(self.train_step_counter.numpy(), avg))
        returns = [avg]

        #Set CTRL+C handler
        signal.signal(signal.SIGINT, LunarLander.handler)


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

        print("Start training at {}".format(LunarLander.dt()))

        step = self.agent.train_step_counter.numpy()
        while step < self.num_iterations:

            if LunarLander.bFinishTrain:
                print("Finish flag detected")
                break

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
                print('step = {0}: loss = {1:0.2f}'.format(step, train_loss))

            if step % self.eval_interval == 0:
                avg = self.compute_avg_return(self._tf_env_eval, self.agent.policy, self.num_eval_episodes)
                print('step = {0}: Average Return = {1:0.2f}'.format(step, avg))
                returns.append(avg)

        print("Finish training at {}".format(LunarLander.dt()))

        if self.is_debug:
            print_summary(self.q_net)

        #save state
        self.save_model(self.agent)
        return returns

    def compute_avg_return(self, environment, policy, num_episodes=10):
        """"""

        print("Start compute average at {}".format(LunarLander.dt()))

        total_return = 0.0
        for eps in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0
            steps = 0
            episod_info = []

            if self.is_debug:
                step_res = [0] + time_step.observation.numpy()[0].tolist() + [time_step.step_type.numpy()[0], time_step.reward.numpy()[0], 0, 0]

            episod_info.append(step_res)
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
                steps += 1

                if self.is_debug:
                    step_res = [steps] + time_step.observation.numpy()[0].tolist() + [time_step.step_type.numpy()[0], time_step.reward.numpy()[0], action_step.action.numpy()[0], int(time_step.is_last())]
                    episod_info.append(step_res)

            if self.is_debug:
                self.print_info(episod_info, eps)

            total_return += episode_return
            print('Episode: {0} return: {1:0.2f} steps {2}'.format(eps, episode_return.numpy()[0], steps))

        print("Finished compute average at {}".format(LunarLander.dt()))

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def print_info(self, data:list, episode:int) -> bool:
        """Print episode information to CSV format"""
        if not self.is_debug:
            return False

        headers = ['Nm','O1','O2','O3','O4','O5','O6','O7','O8','StT','Rwd','Act','Last']
        csv_file = '{0}{1}_{2}.csv'.format(self.debug_data, self.agent.train_step_counter.numpy(), episode)
        with open(csv_file, "w") as fd_write:
            fd_write.write(",".join(headers)+'\n')
            for ln in data:
                fd_write.write(",".join(["{:0.2f}".format(l) for l in ln])+'\n')
            fd_write.close()

        return True

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
        input_lr = tf.keras.layers.Dense(self.observations, activation=None)
        work_layers = [self.gen_layer(lyer_prm[0], lyer_prm[1]) for lyer_prm in self.ly_params]

        """
        Output layer - number od units equal number of actions (4 in our case)
        """
        q_values_layer = tf.keras.layers.Dense(
            self.num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))

        layers = [input_lr] + work_layers + [q_values_layer]
        self.q_net = sequential.Sequential(layers)

        #rint("Q Net Input Spec: {}".format(self.q_net.input_tensor_spec))
        #print("Q Net State Spec: {}".format(self.q_net.state_spec))

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

        #polycy saver
        #self.policy_saver =  PolicySaver(self.agent.collect_policy, batch_size=None)

        self.generation_policy()

    def gen_layer(self, num_units : int, negative_slope: float = 0.0) -> any:
        """
        VarianceScaling
        With distribution="truncated_normal" or "untruncated_normal",
        samples are drawn from a truncated/untruncated normal distribution with a mean of zero and
        a standard deviation (after truncation, if used) stddev = sqrt(scale / n), where n is:
        number of input units in the weight tensor, if mode="fan_in"
        """
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu(negative_slope=negative_slope),
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0 if num_units <= 10 else 2.0,
                mode='fan_in',
                distribution='truncated_normal')
            )


    def generation_policy(self):
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
    mname = sys.argv[1] if len(sys.argv) > 1 else None
    ll = LunarLander(model_name=mname, num_iterations=200)
    ll.prepare()
    res = ll.train_agent()
    print(res)
