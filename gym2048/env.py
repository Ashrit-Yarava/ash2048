import re
from typing import Any

import numpy as np
import gymnasium as gym
import game2048.environment as e
from gymnasium import spaces


class Game2048(gym.Env):
    def __init__(self, binary=False, seed=None):
        """
        Initialize a new 2048 game environment.

        :param binary: Whether to one hot encode the board. Useful for neural networks.
        :param seed: The random seed to use. Defaults to a computer generated seed.
        """
        super(Game2048, self).__init__()

        self.binary = binary
        self.rng = np.random.default_rng(seed)

        # Create the action space.
        self.action_space = spaces.Discrete(4, seed=self.rng)

        # Create the observation space.
        self.board = e.initialize(self.rng)

        if self.binary:
            self.observation_space = spaces.Box(0, 1, (4, 4, 12), seed=self.rng)
        else:
            self.observation_space = spaces.Box(0, 2048, (4, 4), seed=self.rng)

    def step(self, action):
        """
        Take a step with the given action.

        * Reward of 1 if the game was won.
        * Reward of 0 if the game was lost.
        * Reward of -0.1 if the agent made an invalid move.

        :param action: The action to take: (0: up, 1: down, 2: left, 3: right).
        :return: (observation, reward, terminated, truncated, None)
        """
        reward = 0
        done = False

        if e.game_won(self.board) or e.game_lost(self.board):
            raise Exception("Environment has terminated. Call reset!")

        if action not in e.possible_moves(self.board)[0]:
            reward = 0
            done = True  # The game ends if the board
        else:
            self.board = e.move(self.board, action, self.rng)
            if e.game_won(self.board):
                reward = 1
            else:
                reward = 0

        next_obs = self.board if not self.binary else e.to_onehot(self.board)
        done = e.game_won(self.board) or e.game_lost(self.board) or done

        # Return the observation, reward, terminated, truncated, extra info
        return next_obs, reward, done, False, {"board": np.copy(self.board)}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset the environment.

        :param seed: The new seed to use. Old generator will continue otherwise.
        :param options: N/A
        :return: (observation, None)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = e.initialize(self.rng)
        obs = self.board if not self.binary else e.to_onehot(self.board)
        return obs, {"board": np.copy(self.board)}

    def render(self):
        """
        Print the current board state to the terminal.
        """
        b = np.array(self.board, dtype=np.int32)
        print("--------------------")
        print(" " + (re.sub("[\[\]]", "", np.array2string(b, separator="\t"))))
        print("--------------------")
