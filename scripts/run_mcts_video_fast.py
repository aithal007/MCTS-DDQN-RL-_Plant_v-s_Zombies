import sys
from pathlib import Path
import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from pvz_rl.mcts.hybrid_trainer import MCTS, HybridMCTSDDQNTrainer
from pvz_rl.mcts.simulator_wrapper import SimulatorWrapper
from pvz_rl.sim.env import PvZSimEnv

def main():
    print("Initializing fast MCTS+DDQN for video...")
    
    # We will instantiate a new Trainer but only train 5 episodes so it initializes everything properly.
    trainer = HybridMCTSDDQNTrainer(
        difficulty=1,
        reward_scheme='wave_bonus',
        num_simulations=40,   # Strong MCTS lookahead per step!
        hidden_dim=128,
        num_res_blocks=2,
        lr=5e-4,
        batch_size=64,
        buffer_size=5000,
        epsilon_decay_steps=2000,
        eval_freq=10,
        eval_episodes=0,
        checkpoint_dir='checkpoints/fast_mcts',
        log_dir='runs/fast_mcts',
        seed=42,
    )
    
    print("MCTS Search active... Generating the video simulation!")
    
    env = PvZSimEnv(difficulty=1, render_mode="rgb_array", reward_shaping=False)
    env.reset(seed=42)
    engine = env.unwrapped.engine

    out_path = 'C:\\Users\\Lenovo\\.gemini\\antigravity\\brain\\e55754d7-ea08-4cac-b71c-2394479ccecf\\mcts_ddqn_fighting.mp4'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    fps = 15

    # Run for 1500 ticks (zombies spawn at tick 600!)
    for step in range(1500):
        # Render frame
        frame = env.render()
        if writer is None:
            writer = cv2.VideoWriter(out_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

        # MCTS decision using internal MCTS object
        policy, val = trainer.mcts.search(engine)
        
        # Enforce valid rules
        mask = SimulatorWrapper.get_action_mask(engine.state)
        policy = policy * mask
        if policy.sum() > 0:
            policy /= policy.sum()
            action = np.argmax(policy)
        else:
            action = 0 # Wait

        env.step(action)
        engine = env.unwrapped.engine
        
        if step % 100 == 0:
            print(f"Video Frame {step}/1500 processed...")
            
        if engine.state.game_over or engine.state.victory:
            print(f"Ended at step {step}")
            break

    for _ in range(15):
        writer.write(bgr_frame)
    writer.release()
    print("Done generating MCTS video!")

if __name__ == "__main__":
    main()
