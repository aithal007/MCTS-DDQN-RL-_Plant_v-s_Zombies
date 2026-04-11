"""Train baseline RL agents (PPO, DQN, A2C) with different reward schemes.

Usage:
    python scripts/train_baselines.py
    python scripts/train_baselines.py --algo ppo --reward wave_bonus --timesteps 500000
    python scripts/train_baselines.py --algo a2c --difficulty 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from pvz_rl.rewards.reward_shaping import make_shaped_env


def _make_env(algo: str, reward_scheme: str, difficulty: int, render_mode=None):
    """Create env with appropriate wrapping for the algorithm."""
    use_mask = (algo == "maskable_ppo")
    env = make_shaped_env(
        scheme=reward_scheme,
        difficulty=difficulty,
        render_mode=render_mode,
        use_action_mask=use_mask,
    )
    return Monitor(env)


def _make_agent(algo: str, env, lr: float, seed: int, log_dir: str):
    """Create an SB3 agent."""
    common_kwargs = dict(
        verbose=1,
        device="cpu",
        tensorboard_log=log_dir,
        seed=seed,
    )

    if algo == "ppo":
        return PPO(
            "MlpPolicy", env,
            learning_rate=lr,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            **common_kwargs,
        )

    elif algo == "dqn":
        return DQN(
            "MlpPolicy", env,
            learning_rate=lr,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            **common_kwargs,
        )

    elif algo == "a2c":
        return A2C(
            "MlpPolicy", env,
            learning_rate=lr,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            **common_kwargs,
        )

    elif algo == "maskable_ppo":
        try:
            from sb3_contrib import MaskablePPO
        except ImportError:
            raise ImportError("MaskablePPO requires sb3-contrib")
        return MaskablePPO(
            "MlpPolicy", env,
            learning_rate=lr,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            **common_kwargs,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def main():
    parser = argparse.ArgumentParser(description="Train baseline RL agents on PvZ")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "dqn", "a2c", "maskable_ppo"])
    parser.add_argument("--reward", type=str, default="wave_bonus",
                        choices=["sparse", "kill_penalty", "wave_bonus", "time_penalty", "potential"])
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    # Directories
    run_name = f"{args.algo}_{args.reward}_d{args.difficulty}_s{args.seed}"
    ckpt_dir = f"checkpoints/{run_name}"
    log_dir = f"runs/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 60)
    print(f"  BASELINE TRAINING: {args.algo.upper()}")
    print("=" * 60)
    print(f"  Reward:     {args.reward}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Timesteps:  {args.timesteps}")
    print(f"  LR:         {args.lr}")
    print(f"  Seed:       {args.seed}")
    print("=" * 60)

    # Create environments
    env = _make_env(args.algo, args.reward, args.difficulty)
    eval_env = _make_env(args.algo, args.reward, args.difficulty)

    print(f"  Obs space:  {env.observation_space}")
    print(f"  Act space:  {env.action_space}")
    print()

    # Create agent
    model = _make_agent(args.algo, env, args.lr, args.seed, log_dir)

    # Callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=50_000,
            save_path=ckpt_dir,
            name_prefix=run_name,
        ),
        EvalCallback(
            eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            best_model_save_path=os.path.join(ckpt_dir, "best"),
            deterministic=True,
            verbose=1,
        ),
    ]

    # Train
    print("Starting training...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    # Save final
    final_path = os.path.join(ckpt_dir, f"{run_name}_final")
    model.save(final_path)
    print(f"\nModel saved to {final_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
