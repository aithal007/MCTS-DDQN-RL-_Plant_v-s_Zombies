"""Training script for PvZ-RL.

Usage:
    python scripts/train.py --config configs/train_sim.yaml
    python scripts/train.py --config configs/train_steam.yaml
    python scripts/train.py --algo ppo --difficulty 3 --timesteps 200000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

# Ensure the project root is on the path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from pvz_rl.agents.factory import make_agent, make_sim_env


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _make_env(cfg: dict):
    env_cfg = cfg["environment"]
    env_type = env_cfg.get("type", "sim")

    if env_type == "sim":
        env = make_sim_env(
            difficulty=env_cfg.get("difficulty", 1),
            render_mode=env_cfg.get("render_mode"),
            use_action_mask=env_cfg.get("use_action_mask", True),
        )
    elif env_type == "steam":
        from pvz_rl.steam.env import PvZSteamEnv

        env = PvZSteamEnv(
            render_mode=env_cfg.get("render_mode", "rgb_array"),
            obs_width=env_cfg.get("obs_width", 160),
            obs_height=env_cfg.get("obs_height", 120),
            step_delay=env_cfg.get("step_delay", 0.15),
            auto_collect_sun=env_cfg.get("auto_collect_sun", True),
            tray_slots=env_cfg.get("tray_slots", 7),
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

    return Monitor(env)


class CurriculumCallback:
    """Simple callback that bumps difficulty at specified timestep thresholds."""

    def __init__(self, stages: list[dict], env):
        self.stages = sorted(stages, key=lambda s: s["timesteps"])
        self.env = env
        self._current_stage = 0

    def check(self, num_timesteps: int):
        while (
            self._current_stage < len(self.stages) - 1
            and num_timesteps >= self.stages[self._current_stage + 1]["timesteps"]
        ):
            self._current_stage += 1
            new_diff = self.stages[self._current_stage]["difficulty"]
            print(f"\n[Curriculum] Advancing to difficulty {new_diff} at step {num_timesteps}")
            # Update the underlying sim env difficulty
            unwrapped = self.env
            while hasattr(unwrapped, "env"):
                unwrapped = unwrapped.env
            if hasattr(unwrapped, "difficulty"):
                unwrapped.difficulty = new_diff
                unwrapped.engine.difficulty = new_diff


def main():
    parser = argparse.ArgumentParser(description="Train a PvZ-RL agent")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--algo", type=str, default=None, help="Algorithm override")
    parser.add_argument("--difficulty", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = _load_config(args.config)
    else:
        cfg = {
            "environment": {"type": "sim", "difficulty": 1, "render_mode": None, "use_action_mask": True},
            "agent": {"algorithm": "ppo", "device": "auto", "policy_kwargs": {}},
            "training": {
                "total_timesteps": 500_000,
                "eval_freq": 10_000,
                "n_eval_episodes": 5,
                "save_freq": 50_000,
                "checkpoint_dir": "checkpoints",
                "tensorboard_log": "runs",
                "seed": 42,
            },
            "curriculum": {"enabled": False},
        }

    # CLI overrides
    if args.algo:
        cfg["agent"]["algorithm"] = args.algo
    if args.difficulty is not None:
        cfg["environment"]["difficulty"] = args.difficulty
    if args.timesteps is not None:
        cfg["training"]["total_timesteps"] = args.timesteps
    if args.device:
        cfg["agent"]["device"] = args.device
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

    train_cfg = cfg["training"]
    agent_cfg = cfg["agent"]

    # Create environment
    env = _make_env(cfg)
    eval_env = _make_env(cfg)

    print(f"Environment: {cfg['environment']['type']}")
    print(f"Algorithm:   {agent_cfg['algorithm']}")
    print(f"Timesteps:   {train_cfg['total_timesteps']}")
    print(f"Obs space:   {env.observation_space}")
    print(f"Act space:   {env.action_space}")
    print()

    # Create agent
    model = make_agent(
        algorithm=agent_cfg["algorithm"],
        env=env,
        device=agent_cfg.get("device", "auto"),
        tensorboard_log=train_cfg.get("tensorboard_log", "runs"),
        seed=train_cfg.get("seed"),
    )

    # Callbacks
    callbacks = []

    ckpt_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks.append(
        CheckpointCallback(
            save_freq=train_cfg.get("save_freq", 50_000),
            save_path=ckpt_dir,
            name_prefix="pvz_rl",
        )
    )

    callbacks.append(
        EvalCallback(
            eval_env,
            eval_freq=train_cfg.get("eval_freq", 10_000),
            n_eval_episodes=train_cfg.get("n_eval_episodes", 5),
            best_model_save_path=os.path.join(ckpt_dir, "best"),
            deterministic=True,
            verbose=1,
        )
    )

    # Curriculum learning
    curriculum_cfg = cfg.get("curriculum", {})
    curriculum = None
    if curriculum_cfg.get("enabled"):
        curriculum = CurriculumCallback(curriculum_cfg["stages"], env)

    # Train!
    print("Starting training...")
    try:
        if curriculum:
            # Manual training loop for curriculum support
            total = train_cfg["total_timesteps"]
            chunk = min(train_cfg.get("eval_freq", 10_000), total)
            timesteps_done = 0
            model.learn(total_timesteps=chunk, callback=CallbackList(callbacks), reset_num_timesteps=True)
            timesteps_done += chunk
            while timesteps_done < total:
                curriculum.check(timesteps_done)
                remaining = min(chunk, total - timesteps_done)
                model.learn(total_timesteps=remaining, callback=CallbackList(callbacks), reset_num_timesteps=False)
                timesteps_done += remaining
        else:
            model.learn(
                total_timesteps=train_cfg["total_timesteps"],
                callback=CallbackList(callbacks),
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    # Save final model
    final_path = os.path.join(ckpt_dir, "pvz_rl_final")
    model.save(final_path)
    print(f"\nModel saved to {final_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
