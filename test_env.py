import tensorflow as tf
import gym


from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.specs import tensor_spec


def test_env() -> None:
    """Test Enviromnent creating"""
    print("Version 0.3")
    py_env = suite_gym.load("LunarLander-v2")
    #py_env_gym = gym.make("LunarLander-v3")
    #py_env = suite_gym.wrap_env(gym_env=py_env_gym)

    time_step = py_env.reset()

    print('Py Env Time step: {}'.format(time_step))

    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    tf_time_step = tf_env.reset()

    print('TF Env Time step {}'.format(tf_time_step))

    observations = tf_env.time_step_spec().observation.shape[0]

    print('Time Step Spec: {}'.format(tf_env.time_step_spec()))
    print('Observation Num: {} Spec: {}'.format(observations, tf_env.time_step_spec().observation))
    print('Reward Spec: {}'.format(tf_env.time_step_spec().reward))
    print('Action Spec: {}'.format(tf_env.action_spec()))


    action_tensor_spec = tensor_spec.from_spec(tf_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    print('Num actions: {}'.format(num_actions))

if __name__ == '__main__':
    test_env()
