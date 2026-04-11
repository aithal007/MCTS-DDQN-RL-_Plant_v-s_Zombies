"""Analyze and plot results from PvZ-RL experiments.

Generates publication-quality plots:
    - Learning curves (reward vs episodes)
    - Win-rate comparison bar charts
    - Reward breakdown analysis
    - MCTS ablation plots
    - Statistical significance tables

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --results-dir results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _setup_matplotlib():
    """Configure matplotlib for publication-quality plots."""
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 11,
        "figure.dpi": 150,
    })
    return plt


def plot_reward_ablation(results_dir: str, output_dir: str):
    """Plot reward ablation results."""
    plt = _setup_matplotlib()

    fpath = Path(results_dir) / "reward_ablation_all.json"
    if not fpath.exists():
        print(f"  [SKIP] {fpath} not found")
        return

    with open(fpath) as f:
        data = json.load(f)

    # Group by algorithm
    algos = sorted(set(r["algo"] for r in data))
    schemes = ["sparse", "kill_penalty", "wave_bonus", "time_penalty", "potential"]
    scheme_labels = ["Sparse", "Kill/Penalty", "Wave Bonus", "Time Penalty", "Potential"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    fig, axes = plt.subplots(1, len(algos), figsize=(6 * len(algos), 5), sharey=True)
    if len(algos) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        win_rates = []
        stds = []
        for scheme in schemes:
            group = [r["win_rate"] for r in data
                     if r["algo"] == algo and r["reward_scheme"] == scheme]
            win_rates.append(np.mean(group) if group else 0)
            stds.append(np.std(group) if group else 0)

        bars = ax.bar(range(len(schemes)), win_rates, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels(scheme_labels, rotation=30, ha="right", fontsize=10)
        ax.set_title(f"{algo.upper()}", fontweight="bold")
        ax.set_ylabel("Win Rate" if ax == axes[0] else "")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50%")

    plt.suptitle("Reward Shaping Ablation: Win Rate by Scheme", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_ablation_winrate.png"), bbox_inches="tight")
    plt.close()
    print("  Saved: reward_ablation_winrate.png")

    # Also plot average reward
    fig, axes = plt.subplots(1, len(algos), figsize=(6 * len(algos), 5), sharey=True)
    if len(algos) == 1:
        axes = [axes]

    for ax, algo in zip(axes, algos):
        rewards_means = []
        rewards_stds = []
        for scheme in schemes:
            group = [r["avg_reward"] for r in data
                     if r["algo"] == algo and r["reward_scheme"] == scheme]
            rewards_means.append(np.mean(group) if group else 0)
            rewards_stds.append(np.std(group) if group else 0)

        ax.bar(range(len(schemes)), rewards_means, yerr=rewards_stds, capsize=5,
               color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels(scheme_labels, rotation=30, ha="right", fontsize=10)
        ax.set_title(f"{algo.upper()}", fontweight="bold")
        ax.set_ylabel("Avg Episode Reward" if ax == axes[0] else "")

    plt.suptitle("Reward Shaping Ablation: Average Reward", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_ablation_reward.png"), bbox_inches="tight")
    plt.close()
    print("  Saved: reward_ablation_reward.png")


def plot_algorithm_comparison(results_dir: str, output_dir: str):
    """Plot algorithm comparison results."""
    plt = _setup_matplotlib()

    fpath = Path(results_dir) / "algorithm_comparison.json"
    if not fpath.exists():
        print(f"  [SKIP] {fpath} not found")
        return

    with open(fpath) as f:
        data = json.load(f)

    algos = ["ppo", "dqn", "a2c", "mcts_ddqn"]
    algo_labels = ["PPO", "DQN", "A2C", "MCTS+DDQN"]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Win rate
    win_rates = []
    win_stds = []
    for algo in algos:
        group = [r["win_rate"] for r in data if r["algo"] == algo]
        win_rates.append(np.mean(group) if group else 0)
        win_stds.append(np.std(group) if group else 0)

    bars = ax1.bar(range(len(algos)), win_rates, yerr=win_stds, capsize=5,
                   color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(len(algos)))
    ax1.set_xticklabels(algo_labels, fontsize=12)
    ax1.set_ylabel("Win Rate")
    ax1.set_title("Win Rate by Algorithm", fontweight="bold")
    ax1.set_ylim(0, 1.05)

    # Add value labels on bars
    for bar, val in zip(bars, win_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.1%}", ha="center", fontsize=11)

    # Average reward
    rewards_means = []
    rewards_stds = []
    for algo in algos:
        group = [r["avg_reward"] for r in data if r["algo"] == algo]
        rewards_means.append(np.mean(group) if group else 0)
        rewards_stds.append(np.std(group) if group else 0)

    bars = ax2.bar(range(len(algos)), rewards_means, yerr=rewards_stds, capsize=5,
                   color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(len(algos)))
    ax2.set_xticklabels(algo_labels, fontsize=12)
    ax2.set_ylabel("Average Episode Reward")
    ax2.set_title("Average Reward by Algorithm", fontweight="bold")

    plt.suptitle("Algorithm Comparison", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "algorithm_comparison.png"), bbox_inches="tight")
    plt.close()
    print("  Saved: algorithm_comparison.png")


def plot_mcts_ablation(results_dir: str, output_dir: str):
    """Plot MCTS simulation count ablation."""
    plt = _setup_matplotlib()

    fpath = Path(results_dir) / "mcts_ablation.json"
    if not fpath.exists():
        print(f"  [SKIP] {fpath} not found")
        return

    with open(fpath) as f:
        data = json.load(f)

    sim_counts = sorted(set(r["num_simulations"] for r in data))

    win_rates = []
    win_stds = []
    rewards_means = []

    for n in sim_counts:
        group = [r for r in data if r["num_simulations"] == n]
        win_rates.append(np.mean([r["win_rate"] for r in group]))
        win_stds.append(np.std([r["win_rate"] for r in group]))
        rewards_means.append(np.mean([r["avg_reward"] for r in group]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.errorbar(sim_counts, win_rates, yerr=win_stds, marker="o",
                 linewidth=2, markersize=8, capsize=5, color="#9b59b6")
    ax1.set_xlabel("MCTS Simulations per Step")
    ax1.set_ylabel("Win Rate")
    ax1.set_title("Win Rate vs MCTS Simulations", fontweight="bold")
    ax1.set_ylim(0, 1.05)

    ax2.plot(sim_counts, rewards_means, marker="s", linewidth=2, markersize=8, color="#e74c3c")
    ax2.set_xlabel("MCTS Simulations per Step")
    ax2.set_ylabel("Average Episode Reward")
    ax2.set_title("Reward vs MCTS Simulations", fontweight="bold")

    plt.suptitle("MCTS Simulation Count Ablation", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mcts_ablation.png"), bbox_inches="tight")
    plt.close()
    print("  Saved: mcts_ablation.png")


def plot_training_curves(log_dir: str, output_dir: str):
    """Plot training curves from MCTS+DDQN logs."""
    plt = _setup_matplotlib()

    fpath = Path(log_dir) / "mcts_ddqn" / "training_log.json"
    if not fpath.exists():
        print(f"  [SKIP] {fpath} not found")
        return

    with open(fpath) as f:
        data = json.load(f)

    episodes = [d["episode"] for d in data]
    rewards = [d["reward"] for d in data]
    victories = [float(d["victory"]) for d in data]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

    # Reward curve
    ax1.plot(episodes, rewards, alpha=0.3, color="blue", linewidth=0.5)
    if len(rewards) > 20:
        window = min(50, len(rewards) // 5)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax1.plot(episodes[window-1:], smoothed, color="red", linewidth=2, label=f"Smoothed (w={window})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Training Reward Curve (MCTS+DDQN)", fontweight="bold")
    ax1.legend()

    # Win rate curve (rolling window)
    if len(victories) > 20:
        window = min(50, len(victories) // 5)
        rolling_wr = np.convolve(victories, np.ones(window) / window, mode="valid")
        ax2.plot(episodes[window-1:], rolling_wr, color="green", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win Rate (rolling)")
    ax2.set_title("Win Rate Over Training", fontweight="bold")
    ax2.set_ylim(0, 1.05)

    # Losses
    q_losses = [d.get("q_loss", 0) for d in data]
    policy_losses = [d.get("policy_loss", 0) for d in data]
    if any(q_losses):
        ax3.plot(episodes, q_losses, alpha=0.5, color="orange", label="Q-Loss")
    if any(policy_losses):
        ax3.plot(episodes, policy_losses, alpha=0.5, color="purple", label="Policy Loss")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Loss")
    ax3.set_title("Training Losses", fontweight="bold")
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves_mcts_ddqn.png"), bbox_inches="tight")
    plt.close()
    print("  Saved: training_curves_mcts_ddqn.png")


def generate_summary_table(results_dir: str, output_dir: str):
    """Generate a comprehensive results summary table."""
    summary = []
    summary.append("=" * 90)
    summary.append("  PvZ-RL EXPERIMENT RESULTS SUMMARY")
    summary.append("=" * 90)

    for fname in ["reward_ablation_all.json", "algorithm_comparison.json", "mcts_ablation.json"]:
        fpath = Path(results_dir) / fname
        if not fpath.exists():
            continue

        with open(fpath) as f:
            data = json.load(f)

        summary.append(f"\n--- {fname} ---")
        summary.append(f"{'Config':<30} {'Win Rate':>10} {'Avg Reward':>12} {'Avg Kills':>10}")
        summary.append("-" * 70)

        if "reward" in fname or "algorithm" in fname:
            key_field = "algo" if "algorithm" in fname else "reward_scheme"
            configs = sorted(set(
                f"{r.get('algo', 'N/A')}_{r.get('reward_scheme', 'N/A')}" for r in data
            ))
            for cfg in configs:
                parts = cfg.split("_", 1)
                group = [r for r in data
                         if r.get("algo") == parts[0]
                         and (len(parts) < 2 or r.get("reward_scheme", "") == parts[1])]
                if group:
                    wr = np.mean([r["win_rate"] for r in group])
                    rw = np.mean([r["avg_reward"] for r in group])
                    k = np.mean([r.get("avg_kills", 0) for r in group])
                    summary.append(f"{cfg:<30} {wr:>9.1%} {rw:>12.1f} {k:>10.1f}")

        elif "mcts" in fname:
            for n_sims in sorted(set(r["num_simulations"] for r in data)):
                group = [r for r in data if r["num_simulations"] == n_sims]
                wr = np.mean([r["win_rate"] for r in group])
                rw = np.mean([r["avg_reward"] for r in group])
                k = np.mean([r.get("avg_kills", 0) for r in group])
                summary.append(f"MCTS sims={n_sims:<20} {wr:>9.1%} {rw:>12.1f} {k:>10.1f}")

    summary.append("\n" + "=" * 90)
    text = "\n".join(summary)
    print(text)

    with open(os.path.join(output_dir, "results_summary.txt"), "w") as f:
        f.write(text)
    print(f"\n  Saved: results_summary.txt")


def main():
    parser = argparse.ArgumentParser(description="Analyze PvZ-RL experiment results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--output-dir", type=str, default="results/plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  PvZ-RL RESULTS ANALYSIS")
    print("=" * 60)

    plot_reward_ablation(args.results_dir, args.output_dir)
    plot_algorithm_comparison(args.results_dir, args.output_dir)
    plot_mcts_ablation(args.results_dir, args.output_dir)
    plot_training_curves(args.log_dir, args.output_dir)
    generate_summary_table(args.results_dir, args.output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
