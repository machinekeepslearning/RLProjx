from gymnasium.envs.registration import register

register(
    id="asteroids_env/GridWorld-v0",
    entry_point="asteroids_env.envs:GridWorldEnv",
)
