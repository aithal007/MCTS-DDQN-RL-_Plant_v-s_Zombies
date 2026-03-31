"""Evaluate a trained agent or play interactively.

Usage:
    # Watch a trained agent play the simulator
    python scripts/evaluate.py --model checkpoints/pvz_rl_final.zip --render

    # Run N episodes and print stats
    python scripts/evaluate.py --model checkpoints/pvz_rl_final.zip --episodes 20

    # Play the simulator yourself (keyboard controls)
    python scripts/evaluate.py --human
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def evaluate_agent(model_path: str, episodes: int, render: bool, difficulty: int):
    from stable_baselines3 import PPO, DQN

    from pvz_rl.agents.factory import make_sim_env

    env = make_sim_env(
        difficulty=difficulty,
        render_mode="human" if render else None,
        use_action_mask=False,  # standard eval
    )

    # Try loading as PPO first, then DQN
    try:
        model = PPO.load(model_path, env=env)
    except Exception:
        model = DQN.load(model_path, env=env)

    print(f"Evaluating {model_path} for {episodes} episodes (difficulty={difficulty})")

    rewards = []
    victories = 0
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if render:
                env.render()

        rewards.append(total_reward)
        if info.get("victory"):
            victories += 1
        print(f"  Episode {ep + 1}: reward={total_reward:.1f}  victory={info.get('victory', False)}")

    env.close()
    print(f"\nResults over {episodes} episodes:")
    print(f"  Mean reward: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")
    print(f"  Win rate:    {victories}/{episodes} ({100 * victories / episodes:.0f}%)")
    print(f"  Max reward:  {np.max(rewards):.1f}")


def human_play(difficulty: int):
    """Let a human play the simulator with keyboard/mouse controls."""
    import pygame

    from pvz_rl.sim.constants import (
        CELL_H,
        CELL_W,
        COLS,
        FPS,
        GRID_ORIGIN_X,
        GRID_ORIGIN_Y,
        PLANT_NAMES,
        ROWS,
        PlantType,
    )
    from pvz_rl.sim.game import PvZEngine
    from pvz_rl.sim.renderer import PvZRenderer

    engine = PvZEngine(difficulty=difficulty)
    engine.reset()
    renderer = PvZRenderer(headless=False)

    plant_types = [pt for pt in PlantType if pt != PlantType.NONE]
    selected_plant = 0  # index into plant_types

    print("=== PvZ-RL Human Mode ===")
    print("Controls:")
    print("  1-7       : select plant type")
    print("  Click     : place plant on lawn")
    print("  Space     : pause/unpause")
    print("  ESC       : quit")
    print()
    for i, pt in enumerate(plant_types):
        from pvz_rl.sim.constants import PLANT_STATS
        cost = PLANT_STATS[pt]["cost"]
        print(f"  {i + 1}: {PLANT_NAMES[pt]} (cost: {cost})")
    print()

    paused = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif pygame.K_1 <= event.key <= pygame.K_7:
                    idx = event.key - pygame.K_1
                    if idx < len(plant_types):
                        selected_plant = idx
                        print(f"Selected: {PLANT_NAMES[plant_types[selected_plant]]}")
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                col = (mx - GRID_ORIGIN_X) // CELL_W
                row = (my - GRID_ORIGIN_Y) // CELL_H
                if 0 <= row < ROWS and 0 <= col < COLS:
                    pt = plant_types[selected_plant]
                    ok = engine.try_plant(pt, row, col)
                    if ok:
                        print(f"Placed {PLANT_NAMES[pt]} at ({row}, {col})")
                    else:
                        print(f"Cannot place {PLANT_NAMES[pt]} at ({row}, {col})")

        if not paused:
            engine.step()

        renderer.render(engine.state, fps=FPS)

        if engine.state.game_over:
            print("GAME OVER!")
            pygame.time.wait(3000)
            running = False
        elif engine.state.victory:
            print("VICTORY!")
            pygame.time.wait(3000)
            running = False

    renderer.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate or play PvZ-RL")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--human", action="store_true", help="Play as a human")
    args = parser.parse_args()

    if args.human:
        human_play(args.difficulty)
    elif args.model:
        evaluate_agent(args.model, args.episodes, args.render, args.difficulty)
    else:
        print("Specify --model <path> to evaluate, or --human to play yourself.")
        sys.exit(1)


if __name__ == "__main__":
    main()
