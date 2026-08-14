from typing import Any, SupportsFloat
from enum import Enum
import gymnasium as gym
import numpy
import pygame
from gymnasium import spaces
import numpy as np
from gymnasium.core import ObsType, ActType, RenderFrame
from gymnasium_env.env_classes.asteroid_world_classes import *
import threading
import time


class Actions(Enum):
    forward = 0
    backward = 1
    rotate_cw = 2
    rotate_ccw = 3
    fire = 4


def cooldown(cool_time, cooldown_dict, key):
    time.sleep(cool_time)

    cooldown_dict.update({key: True})


class AsteroidEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, bounds=(800, 800), obs_type="vector"):
        self.lives = 10
        self.max_lives = 10

        self.bounds = bounds

        self.asteroids = {}
        self.asteroid_list = list(self.asteroids.values())
        self.asteroid_pos = get_positions(self.asteroid_list)
        self.asteroid_radii = get_radii(self.asteroid_list)
        self.asteroid_id = 0

        self.lasers = {}
        self.laser_id = 0
        self.off_cooldown = {
            "Asteroid": True,
            "Laser": True,
        }
        self.laser_speed = 200

        self.reward = 0
        self.bot = Agent(radius=20,
                         speed=70,
                         angular_speed=2,
                         x=bounds[0] / 2, y=bounds[0] / 2,
                         color="blue", bounds=bounds)

        #Initialize Sensors
        self.num_sensors = 50
        self.sensor_unit_vectors = numpy.zeros((self.num_sensors, 2))
        self.start_sensor_angles = numpy.linspace(0, self.num_sensors - 1, self.num_sensors) * (
                2 * math.pi) / self.num_sensors
        self.sensor_unit_vectors[:, 0] = numpy.cos(self.start_sensor_angles)
        self.sensor_unit_vectors[:, 1] = numpy.sin(self.start_sensor_angles)
        self.min_along = []
        self.sensor_color = ["green"] * self.num_sensors
        self.sensor_color[0] = "red"
        self.render_sensors = False

        self.obs_type = obs_type
        # Gymnasium Variables
        # For surface arrays, pygame outputs as W H C but torch conv2d accepts only C H W
        if self.obs_type == "vector":
            self.observation_space = spaces.Box(0, 1, shape=(self.num_sensors,), dtype=numpy.float32)
        elif self.obs_type == "rgb":
            self.observation_space = spaces.Box(0, 255, shape=(3, bounds[1], bounds[0]), dtype=numpy.uint8)
        elif self.obs_type == "grayscale":
            self.observation_space = spaces.Box(0, 255, shape=(1, bounds[1], bounds[0]), dtype=numpy.uint8)
        self.action_space = spaces.Discrete(5)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        if self.render_mode == "human" and self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(bounds)
            self.window.fill("black")
        elif self.render_mode == "rgb_array" or self.obs_type == "rgb" or self.obs_type == "grayscale":
            pygame.init()
            self.window = pygame.Surface(bounds)
            self.window.fill("black")

    def close(self):
        if pygame.get_init():
            pygame.quit()

    def _spawn_asteroid(self, min_speed, max_speed, min_rad, max_rad, max_roids):
        if len(self.asteroids) < max_roids and self.off_cooldown["Asteroid"]:
            self.off_cooldown.update({"Asteroid": False})
            self.asteroids.update({self.asteroid_id: Asteroid(
                speed=random.randint(min_speed, max_speed),
                radius=random.randint(min_rad, max_rad),
                id=self.asteroid_id, bounds=self.bounds)})
            self.asteroid_id += 1

            threading.Thread(target=cooldown, args=(1, self.off_cooldown, "Asteroid")).start()

    def _perform_action(self, action):
        if action < 5:
            action = Actions(action)
        match action:
            case Actions.forward:
                self.bot.center += self.bot.velocity
            case Actions.backward:
                self.bot.center -= self.bot.velocity
            case Actions.rotate_cw:
                self.bot.angle += self.bot.angular_speed
            case Actions.rotate_ccw:
                self.bot.angle -= self.bot.angular_speed
            case Actions.fire:
                if self.off_cooldown["Laser"]:
                    self.off_cooldown.update({"Laser": False})
                    self.lasers.update({self.laser_id: Laser(self.bot.center[0], self.bot.center[1],
                                                             self.bot.angle, self.laser_speed,
                                                             self.bot.laser_rad, self.laser_id, "yellow", self.bounds)})
                    self.laser_id += 1

                    threading.Thread(target=cooldown, args=(0.4, self.off_cooldown, "Laser")).start()

    def _update_bot(self, action):
        new_sensor_angles = self.bot.angle + self.start_sensor_angles
        self.sensor_unit_vectors = numpy.zeros_like(self.sensor_unit_vectors)
        self.sensor_unit_vectors[:, 0] = numpy.cos(new_sensor_angles)
        self.sensor_unit_vectors[:, 1] = numpy.sin(new_sensor_angles)
        self._perform_action(action)
        self.bot.update()

    def _check_collisions(self):
        laser_keys = list(self.lasers.keys())
        asteroid_keys = list(self.asteroids.keys())
        if len(self.lasers) > 0:
            laser_list = list(self.lasers.values())
            laser_pos = get_positions(laser_list)
            la_disp = numpy.zeros((len(self.asteroids), len(self.lasers), 2))
            for i in range(len(self.asteroids)):
                la_disp[i] = self.asteroid_pos[i] - laser_pos
            la_collisions = numpy.linalg.norm(la_disp, axis=2) < (self.asteroid_radii + self.bot.laser_rad)
            laser_collisions = numpy.any(la_collisions, axis=0)
            asteroid_collisions = numpy.any(la_collisions, axis=1)

            # Perform deletions
            for i in range(len(self.lasers)):
                if laser_collisions[i] == 1:
                    self.lasers.pop(laser_keys[i], None)
                    self.reward += 0.5
                elif is_out_of_bounds(laser_pos[i], self.bounds):
                    self.lasers.pop(laser_keys[i], None)
            for i in range(len(self.asteroids)):
                if asteroid_collisions[i] == 1:
                    self.asteroids.pop(asteroid_keys[i], None)

        if len(self.asteroids) > 0:
            ba_disp = self.asteroid_pos - self.bot.center
            ba_dist = numpy.linalg.norm(ba_disp, axis=1)
            ba_collisions = (self.asteroid_radii + self.bot.radius).flatten() > ba_dist
            for i in range(len(self.asteroids)):
                if ba_collisions[i] == 1:
                    self.asteroids.pop(asteroid_keys[i], None)
                    self.reward -= 0.6
                    self.lives -= 1
                elif is_out_of_bounds(self.asteroid_pos[i], self.bounds, 300):
                    self.asteroids.pop(asteroid_keys[i], None)

        if is_out_of_bounds(self.bot.center, self.bounds):
            self.reward -= 0.6
            self.lives -= 1
            self.bot.center = self.bot.start_pos.copy()

    def reset(self, seed=None, options=None):
        self.lives = self.max_lives

        asteroid_keys = list(self.asteroids.keys())
        for i in range(len(asteroid_keys)):
            self.asteroids.pop(asteroid_keys[i], None)

        laser_keys = list(self.lasers.keys())
        for i in range(len(laser_keys)):
            self.lasers.pop(laser_keys[i], None)

        self.bot.reset()

        inputs = self._get_obs()
        return inputs, {"reward": numpy.array([0], dtype=numpy.float32)}

    def _find_sensors_coll(self):
        # Sensing
        rel_asteroid_pos = self.asteroid_pos - self.bot.center
        dist = numpy.linalg.norm(rel_asteroid_pos, axis=1)
        dist = numpy.expand_dims(dist, -1)
        # Projection/Normals: axis 0: Asteroids, axis 1: Sensors
        projections = rel_asteroid_pos.dot(self.sensor_unit_vectors.transpose())
        normals = numpy.sqrt(numpy.square(dist.repeat(self.num_sensors, axis=1)) - numpy.square(projections))
        along_sensor = projections - numpy.sqrt(
            numpy.square(self.asteroid_radii.repeat(self.num_sensors, axis=1)) - numpy.square(normals))

        # Compute and stack border checks
        border_normals = numpy.array([[0, 1],
                                      [-1, 0],
                                      [0, -1],
                                      [1, 0]])
        border_starts = numpy.array([[0, 0],
                                     [0, 0],
                                     [self.bounds[0], self.bounds[1]],
                                     [self.bounds[0], self.bounds[1]]])
        delta = border_starts - self.bot.center
        numer = numpy.expand_dims(numpy.sum(numpy.multiply(border_normals, delta), axis=1), -1)
        denom = numpy.matmul(border_normals, self.sensor_unit_vectors.transpose())
        border_dist = numer / denom

        # Combine and Find closest
        combined = numpy.vstack((along_sensor, border_dist))
        check_invalid = numpy.logical_and(numpy.logical_not(numpy.isnan(combined)), combined > 0)

        self.min_along = numpy.min(combined, where=check_invalid, axis=0, initial=1000) - self.bot.radius

        return (self.min_along.flatten() / 1000).astype(np.float32)

    def _get_obs(self):
        if self.obs_type == "vector":
            return self._find_sensors_coll()
        elif self.obs_type == "rgb":
            rgb = numpy.array(pygame.surfarray.array3d(self.window), dtype=numpy.uint8)
            rgb = numpy.swapaxes(rgb, 0, 2)
            return rgb
        elif self.obs_type == "grayscale":
            gray_scaled = numpy.mean(pygame.surfarray.array3d(self.window), axis=-1, dtype=numpy.uint8, keepdims=True)
            gray_scaled = numpy.swapaxes(gray_scaled, 0, 2)
            return gray_scaled

    def step(self, action):
        self.reward = 0
        self._spawn_asteroid(70, 80, 30, 60, 20)

        self.asteroid_list = list(self.asteroids.values())
        self.asteroid_pos = get_positions(self.asteroid_list)
        self.asteroid_radii = get_radii(self.asteroid_list)
        self._update_bot(action)
        for roid in list(self.asteroid_list):
            roid.update()
        for laser in list(self.lasers.values()):
            laser.update()

        observation = self._get_obs()

        self._check_collisions()

        terminated = True if self.lives < 1 else False

        if pygame.get_init():
            self._render_frame()

        return observation, self.reward, terminated, False, {"reward": numpy.array([self.reward], dtype=numpy.float32)}

    def _render_frame(self):
        self.window.fill("black")

        for roid in list(self.asteroids.values()):
            roid.render(self.window)
        for laser in list(self.lasers.values()):
            laser.render(self.window)
        self.bot.render(self.window)

        if self.render_sensors:
            casts = numpy.expand_dims(self.min_along, -1) * self.sensor_unit_vectors
            for i in range(len(casts)):
                start = self.bot.center + self.bot.radius * self.sensor_unit_vectors[i]
                pygame.draw.line(self.window, self.sensor_color[i],
                                 start,
                                 casts[i] + start)

        pygame.event.pump()
        if self.render_mode == "human":
            pygame.display.update()

    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.window)
