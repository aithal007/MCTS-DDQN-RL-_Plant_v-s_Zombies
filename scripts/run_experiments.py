"""Master experiment runner for all PvZ-RL experiments.

Orchestrates the full experimental plan:
    1. Reward ablation (5 schemes × 3 seeds × PPO)
    2. Algorithm comparison (PPO, DQN, A2C, MCTS+DDQN × best reward)
    3. MCTS ablation (vary simulation count: 10, 25, 50, 100)
    4. Statistical analysis

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --phase reward
    python scripts/run_experiments.py --phase algorithm
    python scripts/run_experiments.py --phase mcts
    python scripts/run_experiments.py --phase all
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


def run_reward_ablation(timesteps: int = 200_000, seeds: int = 3):
    """Phase 1: Compare reward schemes."""
    print("\n" + "=" * 70)
    print("  PHASE 1: REWARD ABLATION")
    print("=" * 70)

    from scripts.train_reward_ablation import _train_single, REWARD_SCHEMES

    results = []
    for algo in ["ppo", "dqn", "a2c"]:
        for scheme in REWARD_SCHEMES:
            for seed in range(seeds):
                result = _train_single(
                    algo=algo, reward_scheme=scheme,
                    difficulty=1, timesteps=timesteps,
                    seed=42 + seed, eval_episodes=20,
                )
                results.append(result)

    os.makedirs("results", exist_ok=True)
    with open("results/reward_ablation_all.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_algorithm_comparison(reward_scheme: str = "wave_bonus",
                              timesteps: int = 300_000, seeds: int = 3):
    """Phase 2: Compare algorithms with best reward."""
    print("\n" + "=" * 70)
    print("  PHASE 2: ALGORITHM COMPARISON")
    print("=" * 70)

    from scripts.train_reward_ablation import _train_single

    results = []

    # Baseline agents
    for algo in ["ppo", "dqn", "a2c"]:
        for seed in range(seeds):
            result = _train_single(
                algo=algo, reward_scheme=reward_scheme,
                difficulty=1, timesteps=timesteps,
                seed=42 + seed, eval_episodes=20,
            )
            results.append(result)

    # MCTS+DDQN
    from pvz_rl.mcts.hybrid_trainer import HybridMCTSDDQNTrainer

    for seed in range(seeds):
        print(f"\n--- MCTS+DDQN seed={42+seed} ---")
        trainer = HybridMCTSDDQNTrainer(
            difficulty=1,
            reward_scheme=reward_scheme,
            num_simulations=50,
            lr=1e-4,
            seed=42 + seed,
            eval_freq=50,
            eval_episodes=20,
            checkpoint_dir=f"checkpoints/algo_cmp/mcts_ddqn_s{42+seed}",
            log_dir=f"runs/algo_cmp/mcts_ddqn_s{42+seed}",
        )
        # Convert timesteps to approximate episodes
        # ~1000 steps/episode → episodes ≈ timesteps / 1000
        num_eps = max(timesteps // 1000, 100)
        trainer.train(num_episodes=num_eps)

        # Use final eval
        if trainer.eval_log:
            last_eval = trainer.eval_log[-1]
            results.append({
                "algo": "mcts_ddqn",
                "reward_scheme": reward_scheme,
                "difficulty": 1,
                "seed": 42 + seed,
                "timesteps": trainer.total_steps,
                "win_rate": last_eval["win_rate"],
                "avg_reward": last_eval["avg_reward"],
                "avg_kills": last_eval["avg_kills"],
            })

    os.makedirs("results", exist_ok=True)
    with open("results/algorithm_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def run_mcts_ablation(reward_scheme: str = "wave_bonus", seeds: int = 2):
    """Phase 3: Vary MCTS simulation count."""
    print("\n" + "=" * 70)
    print("  PHASE 3: MCTS ABLATION (simulation count)")
    print("=" * 70)

    from pvz_rl.mcts.hybrid_trainer import HybridMCTSDDQNTrainer

    sim_counts = [0, 10, 25, 50, 100]
    results = []

    for n_sims in sim_counts:
        for seed in range(seeds):
            print(f"\n--- MCTS sims={n_sims}, seed={42+seed} ---")
            trainer = HybridMCTSDDQNTrainer(
                difficulty=1,
                reward_scheme=reward_scheme,
                num_simulations=n_sims,
                lr=1e-4,
                seed=42 + seed,
                eval_freq=50,
                eval_episodes=20,
                checkpoint_dir=f"checkpoints/mcts_abl/sims{n_sims}_s{42+seed}",
                log_dir=f"runs/mcts_abl/sims{n_sims}_s{42+seed}",
            )
            trainer.train(num_episodes=300)

            if trainer.eval_log:
                last_eval = trainer.eval_log[-1]
                results.append({
                    "num_simulations": n_sims,
                    "seed": 42 + seed,
                    "total_steps": trainer.total_steps,
                    "win_rate": last_eval["win_rate"],
                    "avg_reward": last_eval["avg_reward"],
                    "avg_kills": last_eval["avg_kills"],
                })

    os.makedirs("results", exist_ok=True)
    with open("results/mcts_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print(f"{'MCTS Sims':>10} {'Win Rate':>10} {'Avg Reward':>12}")
    print("-" * 60)
    for n_sims in sim_counts:
        sr = [r for r in results if r["num_simulations"] == n_sims]
        if sr:
            print(f"{n_sims:>10} {np.mean([r['win_rate'] for r in sr]):>9.1%} "
                  f"{np.mean([r['avg_reward'] for r in sr]):>12.1f}")

    return results


def statistical_analysis():
    """Phase 4: Statistical significance tests."""
    print("\n" + "=" * 70)
    print("  PHASE 4: STATISTICAL ANALYSIS")
    print("=" * 70)

    from scipy import stats

    # Load results
    results_dir = Path("results")
    analyses = {}

    for fname in ["reward_ablation_all.json", "algorithm_comparison.json", "mcts_ablation.json"]:
        fpath = results_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                data = json.load(f)

            # Group by key variable
            if "reward" in fname:
                groups = {}
                for r in data:
                    key = f"{r['algo']}_{r['reward_scheme']}"
                    groups.setdefault(key, []).append(r["win_rate"])

                # Pairwise t-tests (sparse vs each shaped reward)
                print(f"\n  {fname}:")
                for algo in ["ppo", "dqn", "a2c"]:
                    baseline = groups.get(f"{algo}_sparse", [])
                    for scheme in ["kill_penalty", "wave_bonus", "time_penalty", "potential"]:
                        shaped = groups.get(f"{algo}_{scheme}", [])
                        if len(baseline) >= 2 and len(shaped) >= 2:
                            t_stat, p_val = stats.ttest_ind(shaped, baseline)
                            sig = "*" if p_val < 0.05 else ""
                            print(f"    {algo} sparse vs {scheme}: "
                                  f"t={t_stat:.2f}, p={p_val:.3f} {sig}")

            elif "algorithm" in fname:
                groups = {}
                for r in data:
                    groups.setdefault(r["algo"], []).append(r["win_rate"])

                print(f"\n  {fname}:")
                algos = list(groups.keys())
                for i, a1 in enumerate(algos):
                    for a2 in algos[i+1:]:
                        if len(groups[a1]) >= 2 and len(groups[a2]) >= 2:
                            t_stat, p_val = stats.ttest_ind(groups[a1], groups[a2])
                            sig = "*" if p_val < 0.05 else ""
                            print(f"    {a1} vs {a2}: t={t_stat:.2f}, p={p_val:.3f} {sig}")


def main():
    parser = argparse.ArgumentParser(description="Run all PvZ-RL experiments")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["reward", "algorithm", "mcts", "stats", "all"])
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    if args.phase in ("reward", "all"):
        run_reward_ablation(timesteps=args.timesteps, seeds=args.seeds)

    if args.phase in ("algorithm", "all"):
        run_algorithm_comparison(timesteps=args.timesteps, seeds=args.seeds)

    if args.phase in ("mcts", "all"):
        run_mcts_ablation(seeds=args.seeds)

    if args.phase in ("stats", "all"):
        statistical_analysis()

    print("\n\nAll experiments complete! Run: python scripts/analyze_results.py")


if __name__ == "__main__":
    main()
