"""Visualization utilities for training analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_training_curves(log_dir: str, save_path: Optional[str] = None):
    """Plot reward curves from Monitor CSV logs.

    Args:
        log_dir: Directory containing monitor.csv files.
        save_path: If provided, save figure instead of showing.
    """
    try:
        import matplotlib.pyplot as plt
        from stable_baselines3.common.results_plotter import load_results, ts2xy
    except ImportError:
        print("matplotlib and stable-baselines3 are required for plotting.")
        return

    results = load_results(log_dir)
    x, y = ts2xy(results, "timesteps")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Episode rewards
    ax1.plot(x, y, alpha=0.3, color="blue", label="Episode reward")
    # Smoothed
    if len(y) > 10:
        window = min(50, len(y) // 5)
        smoothed = np.convolve(y, np.ones(window) / window, mode="valid")
        ax1.plot(x[window - 1 :], smoothed, color="red", linewidth=2, label=f"Smoothed (w={window})")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Training Reward Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Episode lengths
    x_len, y_len = ts2xy(results, "timesteps")
    lengths = results["l"].values if "l" in results.columns else []
    if len(lengths) > 0:
        ax2.plot(range(len(lengths)), lengths, alpha=0.3, color="green")
        if len(lengths) > 10:
            window = min(50, len(lengths) // 5)
            smoothed = np.convolve(lengths, np.ones(window) / window, mode="valid")
            ax2.plot(range(window - 1, len(lengths)), smoothed, color="darkgreen", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Length")
    ax2.set_title("Episode Length Over Training")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def record_video(
    model_path: str,
    output_path: str = "pvz_agent.mp4",
    episodes: int = 1,
    difficulty: int = 1,
    fps: int = 30,
):
    """Record a video of the agent playing.

    Requires: opencv-python
    """
    try:
        import cv2
    except ImportError:
        print("opencv-python is required for video recording.")
        return

    from stable_baselines3 import PPO, DQN
    from pvz_rl.agents.factory import make_sim_env

    env = make_sim_env(difficulty=difficulty, render_mode="rgb_array", use_action_mask=False)

    try:
        model = PPO.load(model_path, env=env)
    except Exception:
        model = DQN.load(model_path, env=env)

    writer = None
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            frame = env.render()
            if frame is not None:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                # RGB -> BGR for OpenCV
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if writer:
        writer.release()
        print(f"Video saved to {output_path}")
    env.close()
