import numpy
import keyboard
import matplotlib.pyplot as plt
import torch.backends.mkldnn
import gymnasium as gym
import gymnasium_env
from gymnasium.wrappers import FlattenObservation
from gymnasium_env.helpers.geometry_helpers import *

bounds = (84, 84)

env = gym.make("AsteroidEnv-v0", render_mode="rgb_array", bounds=bounds, obs_type="rgb")

obs, _ = env.reset()

running = True
action = 0

win = pygame.display.set_mode(bounds)
while running:
    pygame.surfarray.blit_array(win, env.render())
    if keyboard.is_pressed("w"):
        action = 0
    elif keyboard.is_pressed("s"):
        action = 1
    elif keyboard.is_pressed("d"):
        action = 2
    elif keyboard.is_pressed("a"):
        action = 3
    elif keyboard.is_pressed("space"):
        action = 4
    else:
        action = 5
    if keyboard.is_pressed("p"):
        running = False
    obs, reward, terminated, truncated, info = env.step(action)
    pygame.display.update()
    if terminated:
        print("GAME OVER")
        env.reset()

env.close()
