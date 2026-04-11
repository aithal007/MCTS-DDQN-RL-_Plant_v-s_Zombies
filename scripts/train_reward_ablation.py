"""Reward ablation study: trains the same algorithm with all 5 reward variants.

Compares reward schemes head-to-head with controlled experiments.

Usage:
    python scripts/train_reward_ablation.py
    python scripts/train_reward_ablation.py --algo ppo --timesteps 200000 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from pvz_rl.rewards.reward_shaping import make_shaped_env


REWARD_SCHEMES = ["sparse", "kill_penalty", "wave_bonus", "time_penalty", "potential"]


def _train_single(algo: str, reward_scheme: str, difficulty: int,
                   timesteps: int, seed: int, eval_episodes: int = 10) -> dict:
    """Train a single agent and return evaluation results."""
    run_name = f"{algo}_{reward_scheme}_d{difficulty}_s{seed}"
    ckpt_dir = f"checkpoints/ablation/{run_name}"
    log_dir = f"runs/ablation/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n--- Training {run_name} ---")

    # Create env
    env = Monitor(make_shaped_env(
        scheme=reward_scheme, difficulty=difficulty, use_action_mask=False,
    ))
    eval_env = Monitor(make_shaped_env(
        scheme=reward_scheme, difficulty=difficulty, use_action_mask=False,
    ))

    # Create agent
    common = dict(device="cpu", verbose=0, seed=seed, tensorboard_log=log_dir)
    if algo == "ppo":
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                     batch_size=256, n_epochs=10, gamma=0.99, **common)
    elif algo == "dqn":
        model = DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=50000,
                     batch_size=64, gamma=0.99, **common)
    elif algo == "a2c":
        model = A2C("MlpPolicy", env, learning_rate=7e-4, n_steps=5,
                     gamma=0.99, **common)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    # Train
    start = time.time()
    eval_cb = EvalCallback(eval_env, eval_freq=max(timesteps // 10, 1000),
                            n_eval_episodes=5, verbose=0,
                            best_model_save_path=ckpt_dir)
    model.learn(total_timesteps=timesteps, callback=eval_cb)
    train_time = time.time() - start

    # Final evaluation
    rewards = []
    victories = 0
    total_kills = 0
    for ep in range(eval_episodes):
        obs, info = eval_env.reset(seed=seed + 99999 + ep)
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_reward += r
        rewards.append(ep_reward)
        if info.get("victory"):
            victories += 1
        total_kills += info.get("zombies_killed", 0)

    env.close()
    eval_env.close()

    result = {
        "algo": algo,
        "reward_scheme": reward_scheme,
        "difficulty": difficulty,
        "seed": seed,
        "timesteps": timesteps,
        "train_time": train_time,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "win_rate": victories / eval_episodes,
        "avg_kills": total_kills / eval_episodes,
    }

    print(f"    Win rate: {result['win_rate']:.1%} | "
          f"Avg reward: {result['avg_reward']:.1f} | "
          f"Time: {train_time:.0f}s")

    return result


def main():
    parser = argparse.ArgumentParser(description="Reward ablation study for PvZ")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "dqn", "a2c"])
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds per config")
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()

    print("=" * 60)
    print("  REWARD ABLATION STUDY")
    print("=" * 60)
    print(f"  Algorithm:  {args.algo}")
    print(f"  Difficulty: {args.difficulty}")
    print(f"  Timesteps:  {args.timesteps}")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Schemes:    {REWARD_SCHEMES}")
    print("=" * 60)

    all_results = []

    for scheme in REWARD_SCHEMES:
        for seed in range(args.seeds):
            result = _train_single(
                algo=args.algo,
                reward_scheme=scheme,
                difficulty=args.difficulty,
                timesteps=args.timesteps,
                seed=42 + seed,
                eval_episodes=args.eval_episodes,
            )
            all_results.append(result)

    # Save results
    output_dir = "results/ablation"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"reward_ablation_{args.algo}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Scheme':<15} {'Win Rate':>10} {'Avg Reward':>12} {'Avg Kills':>10} {'Train Time':>12}")
    print("-" * 80)

    for scheme in REWARD_SCHEMES:
        scheme_results = [r for r in all_results if r["reward_scheme"] == scheme]
        avg_wr = np.mean([r["win_rate"] for r in scheme_results])
        avg_rw = np.mean([r["avg_reward"] for r in scheme_results])
        avg_k = np.mean([r["avg_kills"] for r in scheme_results])
        avg_t = np.mean([r["train_time"] for r in scheme_results])
        print(f"{scheme:<15} {avg_wr:>9.1%} {avg_rw:>12.1f} {avg_k:>10.1f} {avg_t:>11.0f}s")

    print("=" * 80)


if __name__ == "__main__":
    main()
