import random
from typing import Any
import pygame
from PIL import Image
import gymnasium as gym
import numpy as np
from gymnasium import spaces

import gym2048.environment as e


pygame.init()


class Game2048(gym.Env):
    def __init__(self, binary=False, seed=None, display=False, save_animation=None):
        """
        Initialize a new 2048 game environment.

        @param binary: Whether to one hot encode the board. Useful for neural networks.
        @param seed: The random seed to use. Defaults to a computer generated seed.
        @param display: Whether to display the changes in a PyGame window.
        @param save_animation: Whether to save the displayed animation to a gif.
        """
        super(Game2048, self).__init__()

        self.binary = binary
        self.display = display
        self.save_animation = save_animation if self.display else False
        self.animation = []
        self.rng = np.random.default_rng(seed)

        # Create the action space.
        self.action_space = spaces.Discrete(4, seed=self.rng)

        # Create the observation space.
        self.board = e.initialize(self.rng)

        if self.binary:
            self.observation_space = spaces.Box(0, 1, (4, 4, 12), seed=self.rng)
        else:
            self.observation_space = spaces.Box(0, 2048, (4, 4), seed=self.rng)

        if self.display:
            self._initialize_render()

    def step(self, action):
        """
        Take a step with the given action.

        * Reward of 1 if the game was won.
        * Reward of 0 if the game was lost.
        * Reward of -0.1 if the agent made an invalid move.

        @param action: The action to take: (0: up, 1: down, 2: left, 3: right).
        @return: (observation, reward, terminated, truncated, None)
        """
        reward = 0
        done = False

        if e.game_won(self.board) or e.game_lost(self.board):
            raise Exception("Environment has terminated. Call reset!")

        if action not in e.possible_moves(self.board)[0]:
            reward = 0
            done = False  # The game ends if the board
        else:
            self.board = e.move(self.board, action, self.rng)
            if e.game_won(self.board):
                reward = 1
            else:
                reward = 0

        next_obs = self.board if not self.binary else e.to_onehot(self.board)
        done = e.game_won(self.board) or e.game_lost(self.board) or done

        if self.display:
            self.render()

        # Return the observation, reward, terminated, truncated, extra info
        return next_obs, reward, done, False, {"board": np.copy(self.board)}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset the environment.

        @param seed: The new seed to use. Old generator will continue otherwise.
        @param options: N/A
        @return: (observation, None)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = e.initialize(self.rng)
        obs = self.board if not self.binary else e.to_onehot(self.board)
        return obs, {"board": np.copy(self.board)}

    def _draw_grid(self):
        for row in range(4):
            for col in range(4):
                value = self.board[row][col]
                color = self.tile_colors.get(value, self.tile_colors[2048])
                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        col * self.cell_size,
                        row * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    ),
                )
                if value != 0:
                    font = pygame.font.Font(None, 36)
                    text = font.render(str(value), True, (0, 0, 0))
                    text_rect = text.get_rect(
                        center=(
                            col * self.cell_size + self.cell_size / 2,
                            row * self.cell_size + self.cell_size / 2,
                        )
                    )
                    self.screen.blit(text, text_rect)

    def _initialize_render(self):
        self.window_size = (400, 400)  # Width, Height
        self.cell_size = 100

        self.tile_colors = {
            0: (255, 255, 255),
            2: (238, 228, 218),
            4: (237, 224, 200),
            8: (242, 177, 121),
            16: (245, 149, 99),
            32: (246, 124, 95),
            64: (246, 94, 59),
            128: (237, 207, 114),
            256: (237, 204, 97),
            512: (237, 200, 80),
            1024: (237, 197, 63),
            2048: (237, 194, 46),
        }

        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("2048 Game")

    def render(self):
        """
        Print the current board state to the terminal.
        """
        self.screen.fill((255, 255, 255))
        self._draw_grid()
        pygame.display.flip()

        if self.save_animation:
            self.animation.append(
                pygame.surfarray.array3d(pygame.display.get_surface())
            )

    def save_render(self, filename):
        """
        Save the animation if possible. Requires the environment to have been
        initialized with the save_animation flag marked True.

        @param filename: The filename to save the generated gif to.
        """
        if len(self.animation) > 2:
            image_list = [Image.fromarray(arr) for arr in self.animation]
            image_list[0].save(
                filename,
                save_all=True,
                append_images=image_list[1:],
                duration=100,  # Duration between frames in milliseconds
                loop=0,
            )
            self.animation = []


if __name__ == "__main__":
    game = Game2048(display=True, save_animation=False)
    obs, _ = game.reset()
    done = False

    while not done:
        _, _, term, trunc, _ = game.step(random.randint(0, game.action_space.n))
        done = term or trunc

    game.save_render("output.gif")
    pygame.quit()
