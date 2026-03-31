"""Quick-start demo: train for a few minutes and watch the result.

Usage:
    python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from pvz_rl.agents.factory import make_agent, make_sim_env


def main():
    print("=== PvZ-RL Quick Demo ===")
    print("Training PPO on the simulator (difficulty 1) for 50k steps...")
    print("This should take a few minutes on CPU.\n")

    env = make_sim_env(difficulty=1, use_action_mask=False)
    model = make_agent("ppo", env, device="cpu", tensorboard_log=None)

    model.learn(total_timesteps=50_000)

    # Evaluate
    print("\n--- Evaluation (5 episodes) ---")
    rewards = []
    wins = 0
    for ep in range(5):
        obs, info = env.reset()
        total_r = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            total_r += r
            done = terminated or truncated
        rewards.append(total_r)
        if info.get("victory"):
            wins += 1
        print(f"  Episode {ep + 1}: reward={total_r:.1f}, victory={info.get('victory', False)}")

    import numpy as np
    print(f"\nMean reward: {np.mean(rewards):.1f}, Win rate: {wins}/5")

    # Save
    model.save("checkpoints/demo_model")
    print("Model saved to checkpoints/demo_model.zip")
    print("\nTo watch it play:  python scripts/evaluate.py --model checkpoints/demo_model.zip --render")

    env.close()


if __name__ == "__main__":
    main()
