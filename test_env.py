import tensorflow as tf
import gymnasium as gym


from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment


def test_env() -> None:
    """Test Enviromnent creating"""
    py_env = gym.make("LunarLander-v3")
    observation, info = py_env.reset()

    print(observation)
    print(info)

    tf_env = tf_py_environment.TFPyEnvironment(py_env)
    tf_observation, tf_info = tf_env.reset()

    print(tf_observation)
    print(tf_info)


if __name__ == '__main__':
    test_env()
