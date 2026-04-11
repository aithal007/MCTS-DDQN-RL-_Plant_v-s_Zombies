"""Train the hybrid MCTS + DDQN agent on the PvZ simulator.

Usage:
    python scripts/train_mcts.py
    python scripts/train_mcts.py --config configs/train_mcts.yaml
    python scripts/train_mcts.py --difficulty 2 --episodes 1000 --sims 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train MCTS+DDQN agent for PvZ")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--difficulty", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--sims", type=int, default=None, help="MCTS simulations per step")
    parser.add_argument("--reward", type=str, default=None,
                        choices=["sparse", "kill_penalty", "wave_bonus", "time_penalty", "potential"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    args = parser.parse_args()

    # Default config
    cfg = {
        "difficulty": 1,
        "reward_scheme": "wave_bonus",
        "num_episodes": 5000,
        "max_steps_per_episode": 10000,
        "num_simulations": 50,
        "c_puct": 1.5,
        "temperature": 1.0,
        "temperature_decay": 0.995,
        "lr": 1e-4,
        "gamma": 0.99,
        "batch_size": 256,
        "buffer_size": 100_000,
        "target_update_freq": 1000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 50_000,
        "mcts_policy_weight": 1.0,
        "hidden_dim": 256,
        "num_res_blocks": 4,
        "eval_freq": 10,
        "eval_episodes": 10,
        "checkpoint_dir": "checkpoints/mcts_ddqn",
        "log_dir": "runs/mcts_ddqn",
        "seed": 42,
        "device": "cpu",
    }

    # Load from YAML if provided
    if args.config:
        file_cfg = _load_config(args.config)
        cfg.update(file_cfg)

    # CLI overrides
    if args.difficulty is not None:
        cfg["difficulty"] = args.difficulty
    if args.episodes is not None:
        cfg["num_episodes"] = args.episodes
    if args.sims is not None:
        cfg["num_simulations"] = args.sims
    if args.reward is not None:
        cfg["reward_scheme"] = args.reward
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.hidden_dim is not None:
        cfg["hidden_dim"] = args.hidden_dim

    from pvz_rl.mcts.hybrid_trainer import HybridMCTSDDQNTrainer

    trainer = HybridMCTSDDQNTrainer(
        difficulty=cfg["difficulty"],
        reward_scheme=cfg["reward_scheme"],
        hidden_dim=cfg["hidden_dim"],
        num_res_blocks=cfg.get("num_res_blocks", 4),
        num_simulations=cfg["num_simulations"],
        c_puct=cfg["c_puct"],
        temperature=cfg["temperature"],
        temperature_decay=cfg["temperature_decay"],
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        buffer_size=cfg["buffer_size"],
        target_update_freq=cfg["target_update_freq"],
        epsilon_start=cfg["epsilon_start"],
        epsilon_end=cfg["epsilon_end"],
        epsilon_decay_steps=cfg["epsilon_decay_steps"],
        mcts_policy_weight=cfg["mcts_policy_weight"],
        eval_freq=cfg["eval_freq"],
        eval_episodes=cfg["eval_episodes"],
        checkpoint_dir=cfg["checkpoint_dir"],
        log_dir=cfg["log_dir"],
        seed=cfg["seed"],
        device=cfg["device"],
    )

    trainer.train(
        num_episodes=cfg["num_episodes"],
        max_steps_per_episode=cfg["max_steps_per_episode"],
    )


if __name__ == "__main__":
    main()
