"""PvZ-RL: Reinforcement Learning for Plants vs Zombies."""

from gymnasium.envs.registration import register


def register_envs():
    pass


# Register environments with Gymnasium
register(
    id="PvZSim-v0",
    entry_point="pvz_rl.sim.env:PvZSimEnv",
    max_episode_steps=10_000,
)

