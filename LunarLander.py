#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Any, Optional
from xmlrpc.client import Boolean
import gym
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  #'INFO'

import os
import sys
import signal
from datetime import datetime
from datetime import timedelta
import json

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
from tf_agents.policies import random_py_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.drivers import py_driver
from tf_agents.metrics import tf_metrics
from tf_agents.networks.layer_utils import print_summary

"""
{
    "replay_buffer_max_length": 100000,
    "num_eval_episodes": 10,
    "num_iterations": 20000,
    "eval_interval": 100,
    "log_interval": 20,
    "batch_size": 128,
    "layers": [
        {
            "num_units_scale": 4,
            "activation": "gelu"
        },
        {
            "num_units_scale": 2,
            "activation": "relu"
        }
    ]
}
"""

class ModelParams(object):
    """Parameters for model"""

    s_replay_buffer_max_length = 40000
    s_num_eval_episodes = 10
    s_num_iterations = 20000
    s_log_interval = 1000
    s_eval_interval = 2000
    s_batch_size = 2
    s_debug = True
    s_debug_data = 'data/'

    s_layers = "layers"

    def __init__(self) -> None:

        self.mparams = {
            "replay_buffer_max_length" : ModelParams.s_replay_buffer_max_length,
            "num_eval_episodes" : ModelParams.s_num_eval_episodes,
            "num_iterations" : ModelParams.s_num_iterations,
            "log_interval" : ModelParams.s_log_interval, #200
            "eval_interval" : ModelParams.s_eval_interval, #1000
            "batch_size" : ModelParams.s_batch_size,

            "debug" : ModelParams.s_debug,
            "debug_data" : ModelParams.s_debug_data
        }

        self.layers_cfg = [
            {
                "num_units_scale": 1,
                "layer_params": 100,
                "activation": "relu",
                "bias_initializer": -0.2
            },
            {
                "num_units_scale": 1,
                "layer_params": 50,
                "activation": "relu"
            }
        ]

    @property
    def replay_buffer_max_length(self) -> int:
        return self.mparams['replay_buffer_max_length']

    @property
    def num_eval_episodes(self) -> int:
        return self.mparams['num_eval_episodes']

    @property
    def num_iterations(self) -> int:
        return self.mparams['num_iterations']

    @property
    def log_interval(self) -> int:
        return self.mparams['log_interval']

    @property
    def eval_interval(self) -> int:
        return self.mparams['eval_interval']

    @property
    def batch_size(self) -> int:
        return self.mparams['batch_size']

    @property
    def is_debug(self) -> Boolean:
        return self.mparams['debug']

    @property
    def debug_data(self) -> str:
        return self.mparams['debug_data']

    @property
    def layers(self) -> list:
        return self.layers_cfg

    def load_ml_params(self, f_name:str = None) -> bool:
        """Load models parameters"""
        if not f_name:
            return True

        result = True
        self.f_name = f_name
        try:
            fd = open(self.f_name, 'r')
            try:
                ml_parameters = json.load(fd)
                print("Loaded {}".format(f_name))

                """Genaral model parameters"""
                for pkey in self.mparams.keys():
                    if pkey in ml_parameters:
                        self.mparams[pkey] = ml_parameters[pkey]

                """Layers configuration"""

                if ModelParams.s_layers not in ml_parameters or len(ml_parameters[ModelParams.s_layers])==0:
                    print("Invalid/absent layers configuration.")
                    return False

                self.layers_cfg = ml_parameters[ModelParams.s_layers]
                for layer in self.layers_cfg:
                    if "activation" not in layer or layer["activation"] not in ["relu", "gelu"]:
                        layer["activation"] = "relu"

                    if "bias_initializer" in layer and (layer["bias_initializer"] > 1 or  layer["bias_initializer"] < -1):
                        print("Error. Invalid layer configuration. Value for bias_initializer should be between (-1,1).")
                        return False

                    if "layer_params" not in layer:
                        print("Error. Invalid layer configuration. Value for layer_params is absent.")
                        return False

                    if "num_units_scale" not in layer:
                        layer["num_units_scale"] = 1
                    elif layer["num_units_scale"] <= 0 or layer["num_units_scale"] > 20:
                        print("Error. Invalid layer configuration. Value for num_units_scale is incorrect.")
                        return False

            except json.JSONDecodeError as jerr:
                msg = "Error. Could not parse file {} err {} line {}".format(self.f_name, jerr.msg, jerr.lineno)
                print(msg)
                result = False

            fd.close()

        except OSError as err:
            msg = "Error. Could not open file {} errno {} {}".format(self.f_name, err.errno, err.strerror)
            print(msg)
            result = False

        return result


    def to_string(self) -> None:
        """Current configuration"""
        print("replay_buffer_max_length {}\nnum_eval_episodes {}" \
            "\nnum_iterations {}\nlog_interval {}\neval_interval {}\nbatch_size {}".format(
            self.replay_buffer_max_length,
            self.num_eval_episodes,
            self.num_iterations,
            self.log_interval,
            self.eval_interval,
            self.batch_size)
        )


class LunarLander(object):
    """Reinforsment learning for Lunar Lander environment"""

    #Correctly finish train by CTRL+C
    bFinishTrain = False

    @staticmethod
    def handler(signum, frame):
        """Signal processing handler"""
        signame = signal.Signals(signum).name
        print(f'Signal handler called with signal {signame} ({signum})')
        LunarLander.bFinishTrain = True

    @staticmethod
    def dt() -> str:
        return str(datetime.now())

    def __init__(self, cfg_name: Optional [str] = None, model_name: Optional [str] = None) -> None:
        """Initialization"""

        print("Detected GPU: {tf.config.list_physical_devices('GPU')}")

        self.cfg = ModelParams()
        if cfg_name:
            res = self.cfg.load_ml_params(cfg_name)
            print('Load model parameters: {}'.format(res))

        self.model_name = os.path.join(".", model_name) if model_name else None
        self.checkpoint_dir = self.model_name + '.checkpoint' if self.model_name else None
        self.policy_dir = self.model_name + '.policy' if self.model_name else None

        self.py_env = suite_gym.load('LunarLander-v2')
        self.py_env_eval = suite_gym.load('LunarLander-v2')

        self.tf_env = tf_py_environment.TFPyEnvironment(self.py_env)
        self.tf_env_eval = tf_py_environment.TFPyEnvironment(self.py_env_eval)


        action_tensor_spec = self.tf_env.action_spec()
        self.num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        self.observations = self.tf_env.time_step_spec().observation.shape[0]

        self.cfg.to_string()

        if self.is_debug:
            print('Number actions: {}'.format(self.num_actions))
            print('Parameters in Input Layer : {}'.format(self.num_actions))
            print('Observation Num: {}'.format(self.tf_env.time_step_spec().observation.shape[0]))

            print('Time Step Spec: {}'.format(self.tf_env.time_step_spec()))
            print('Observation Spec: {}'.format(self.tf_env.time_step_spec().observation))
            print('Reward Spec: {}'.format(self.tf_env.time_step_spec().reward))
            print('Action Spec: {}'.format(self.tf_env.action_spec()))

    @property
    def mname(self) -> any:
        return self.model_name

    @property
    def is_debug(self) -> bool:
        return self.cfg.is_debug

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

        tm_global_start = datetime.now()

        if self.is_debug:
            print("Counters before. Agent: {} Saved: {}".format(self.agent.train_step_counter, self.train_step_counter))
            print_summary(self.q_net)

        self.agent.train_step_counter.assign(self.train_step_counter)

        avg = self.compute_avg_return(self.tf_env_eval, self.agent.policy, self.cfg.num_eval_episodes)
        returns = [avg]

        if self.is_debug:
            print('step = {0}: Average Return = {1:0.2f}'.format(self.train_step_counter.numpy(), avg))

        #Set CTRL+C handler
        signal.signal(signal.SIGINT, LunarLander.handler)

        step = self.agent.train_step_counter.numpy()
        tm_start = datetime.now()
        self.replay_buffer.clear()

        print("Start training at {} From {} to {}".format(LunarLander.dt(), step, self.cfg.num_iterations))

        while step < self.cfg.num_iterations:  #really there is num of steps

            if LunarLander.bFinishTrain:
                print("Finish flag detected")
                break

            # Collect a few steps and save to the replay buffer.
            self.collect_steps(self.py_env, 2)#, self.agent)

            counter = 0
            while counter < self.replay_buffer.num_frames():
                iterator = iter(self.replay_buffer.as_dataset(sample_batch_size=1))
                trajectories, _ = next(iterator)
                train_loss = self.agent.train(experience=trajectories).loss

                step = self.agent.train_step_counter.numpy()

                if step % self.cfg.log_interval == 0:
                    print('step = {0}: loss = {1:0.2f} Duration {2} sec'.format(step, train_loss, (datetime.now()-tm_start).seconds))
                    tm_start = datetime.now()

                if step % self.cfg.eval_interval == 0:
                    avg = self.compute_avg_return(self.tf_env_eval, self.agent.policy, self.cfg.num_eval_episodes) #self.agent.policy,
                    print('step = {0}: Average Return = {1:0.2f}'.format(step, avg))
                    returns.append(avg)

                counter = counter + 1

                if LunarLander.bFinishTrain:
                    print("Finish flag detected")
                    break

            self.replay_buffer.clear()

        tm_interval = datetime.now() - tm_global_start
        print("Finish training at {} Duration: {}".format(LunarLander.dt(), tm_interval))

        if self.is_debug:
            print_summary(self.q_net)

        #save state
        self.save_model(self.agent)
        return returns

    def collect_steps(self, environment, num_episodes, agent = None):
        """Generate polices"""
        collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            agent.collect_policy, use_tf_function=True) if agent else random_py_policy.RandomPyPolicy(environment.time_step_spec(), environment.action_spec())

        driver = py_driver.PyDriver(environment,
            collect_policy,
            [self.rb_observer],
            max_episodes=num_episodes)

        initial_time_step = environment.reset()
        driver.run(initial_time_step)


    def compute_avg_return(self, environment, policy, num_episodes=10):
        """"""
        print("Start compute average at {}".format(LunarLander.dt()))

        total_return = 0.0
        for eps in range(num_episodes):
            tm_start = datetime.now()
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

            tm_diff = datetime.now() - tm_start
            if self.is_debug:
                self.print_info(episod_info, eps)

            total_return += episode_return
            print('Episode: {0} return: {1:0.2f} steps {2} Duration {3} sec'.format(eps, episode_return.numpy()[0], steps, tm_diff.seconds))

        print("Finished compute average at {}".format(LunarLander.dt()))

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def print_info(self, data:list, episode:int) -> bool:
        """Print episode information to CSV format"""
        if not self.is_debug:
            return False

        if not os.path.exists(self.cfg.debug_data):
            #print("Folder {} does not exist.".format(self.cfg.debug_data))
            return False

        headers = ['Nm','X','Y','Vx','Vy','Angle','Va','LegL','LegR','StT','Reward','Action','Last']
        csv_file = '{0}{1}_{2}_{3}.csv'.format(self.cfg.debug_data,
                                               self.model_name if self.model_name else 'General',
                                               self.agent.train_step_counter.numpy(),
                                               episode)
        with open(csv_file, "w") as fd_write:
            fd_write.write(",".join(headers)+'\n')
            for ln in data:
                fd_write.write(",".join(["{:0.2f}".format(l) for l in ln])+'\n')
            fd_write.close()

        return True

    def model_prepare(self):
        """Generate model
        Load previosly detected weights if needed (not implemented)
        """

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # its output.
        #input_lr = tf.keras.layers.Dense(self.observations, activation=None, name="Input")
        work_layers = [self.gen_layer(lyer_prm) for lyer_prm in self.cfg.layers]

        """
        Output layer - number od units equal number of actions (4 in our case)
        """
        q_values_layer = tf.keras.layers.Dense(
            self.num_actions,
            activation=None,
            name="Output",
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))

        #layers = [input_lr] + work_layers + [q_values_layer]
        #self.q_net = sequential.Sequential(layers=layers, name="LLand")

        layers = work_layers + [q_values_layer]
        self.q_net = sequential.Sequential(layers=layers, input_spec=self.tf_env.time_step_spec().observation, name="LLand")

        self.train_step_counter = tf.Variable(0)
        self.agent = dqn_agent.DqnAgent(
                self.tf_env.time_step_spec(),
                self.tf_env.action_spec(),
                q_network=self.q_net,
                optimizer=tf.keras.optimizers.Adam(), #use lerning rate by default 0.001,
                td_errors_loss_fn=common.element_wise_squared_loss,
                train_step_counter=self.train_step_counter)

        self.agent.initialize()
        self.agent.train = common.function(self.agent.train)

    def gen_layer(self, layer_info) -> any:
        """
        VarianceScaling
        With distribution="truncated_normal" or "untruncated_normal",
        samples are drawn from a truncated/untruncated normal distribution with a mean of zero and
        a standard deviation (after truncation, if used) stddev = sqrt(scale/n), where n is:
        number of input units in the weight tensor, if mode="fan_in"
        """

        num_units = layer_info["num_units_scale"]*layer_info["layer_params"]
        l_name = layer_info["name"] if "name" in layer_info else "{}_{}".format(layer_info["activation"], num_units)

        print("Layer {} parameters:".format(l_name))
        for key, value in layer_info.items():
            print("{} : {}".format(key, value))

        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.gelu if layer_info["activation"] == 'gelu' else tf.keras.activations.relu,
            name=l_name,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=1.0 if num_units <= 10 else 2.0,
                mode='fan_in',
                distribution='truncated_normal'),
            bias_initializer=None if "bias_initializer" not in layer_info else tf.keras.initializers.Constant(layer_info["bias_initializer"])
            )

    def reply_buffer_prepare(self):
        """Prepre replay buffer"""
        replay_buffer_signature = tensor_spec.from_spec(self.agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

        table_name = 'uniform_table'
        table = reverb.Table(
            table_name,
            max_size=self.cfg.replay_buffer_max_length,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=replay_buffer_signature
            )

        self.reverb_server = reverb.Server([table])

        self.replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            table_name=table_name,
            sequence_length=None, #2,
            local_server=self.reverb_server)

        self.rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            self.replay_buffer.py_client,
            table_name,
            sequence_length=2)

if __name__ == '__main__':
    mname = None
    cfgname = None

    if len(sys.argv) == 3:
        cfgname = sys.argv[1]
        mname = sys.argv[2]
    if len(sys.argv) == 2:
        cfgname = sys.argv[1]

    ll = LunarLander(cfg_name=cfgname, model_name=mname)
    ll.prepare()
    res = ll.train_agent()
    print(res)
