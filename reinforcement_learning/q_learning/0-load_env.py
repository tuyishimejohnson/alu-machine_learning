#!/usr/bin/env python3
"""
This module loads the environment.
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym.

    Args:
        desc: Either None or a list of lists containing a custom
              description of the map to load for the environment.
        map_name: Either None or a string containing the pre-made map to load.
        is_slippery: A boolean to determine if the ice is slippery.

    Returns:
        The environment.
    """

    env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name, is_slippery=is_slippery)
    return env
