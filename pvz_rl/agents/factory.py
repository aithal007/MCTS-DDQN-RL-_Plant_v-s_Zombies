"""Pre-configured agent factories for quick experimentation.

Usage:
    from pvz_rl.agents import make_agent
    model = make_agent("ppo", env, device="auto")
    model.learn(total_timesteps=100_000)
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from .feature_extractors import PvZCNNExtractor, PvZSimExtractor
from .wrappers import ActionMaskWrapper, FlattenDictObs


def make_sim_env(
    difficulty: int = 1,
    render_mode: str | None = None,
    use_action_mask: bool = True,
) -> gym.Env:
    """Create a ready-to-train simulator environment."""
    from pvz_rl.sim.env import PvZSimEnv

    env = PvZSimEnv(difficulty=difficulty, render_mode=render_mode)
    env = FlattenDictObs(env)
    if use_action_mask:
        # Note: FlattenDictObs doesn't break get_action_mask because it's on unwrapped
        env = ActionMaskWrapper(env)
    return env


def make_agent(
    algorithm: str,
    env: gym.Env,
    device: str = "auto",
    tensorboard_log: str | None = "./runs",
    **kwargs: Any,
):
    """Create an SB3 agent.

    Args:
        algorithm: One of "ppo", "dqn", "maskable_ppo".
        env: A Gymnasium environment (already wrapped).
        device: "auto", "cpu", or "cuda".
        tensorboard_log: Path for TB logs.
        **kwargs: Extra args forwarded to the SB3 model constructor.

    Returns:
        An SB3 model ready for ``.learn()``.
    """
    algo = algorithm.lower()

    # Determine feature extractor based on obs space
    obs_space = env.observation_space
    policy_kwargs: dict[str, Any] = kwargs.pop("policy_kwargs", {})
    if len(obs_space.shape) == 1:
        # Flat vector -> MLP extractor
        policy_kwargs.setdefault("features_extractor_class", PvZSimExtractor)
        policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 256})
        policy_kwargs.setdefault("net_arch", [256, 128])
    elif len(obs_space.shape) == 3:
        # Image -> CNN extractor
        policy_kwargs.setdefault("features_extractor_class", PvZCNNExtractor)
        policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 256})

    if algo == "ppo":
        from stable_baselines3 import PPO

        return PPO(
            "MlpPolicy" if len(obs_space.shape) == 1 else "CnnPolicy",
            env,
            device=device,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            **kwargs,
        )

    elif algo == "dqn":
        from stable_baselines3 import DQN

        return DQN(
            "MlpPolicy" if len(obs_space.shape) == 1 else "CnnPolicy",
            env,
            device=device,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            verbose=1,
            **kwargs,
        )

    elif algo == "maskable_ppo":
        try:
            from sb3_contrib import MaskablePPO
        except ImportError:
            raise ImportError(
                "MaskablePPO requires sb3-contrib. Install with: pip install sb3-contrib"
            )
        return MaskablePPO(
            "MlpPolicy" if len(obs_space.shape) == 1 else "CnnPolicy",
            env,
            device=device,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Choose from: ppo, dqn, maskable_ppo")
