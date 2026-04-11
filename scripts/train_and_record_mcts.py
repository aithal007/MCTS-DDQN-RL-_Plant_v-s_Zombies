import sys
from pathlib import Path
import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from pvz_rl.mcts.hybrid_trainer import HybridMCTSDDQNTrainer
from pvz_rl.mcts.mcts import MCTS
from pvz_rl.mcts.simulator_wrapper import SimulatorWrapper
from pvz_rl.sim.game import PvZEngine
from pvz_rl.rewards.reward_shaping import RewardConfig, RewardScheme, RewardShaperWrapper

def main():
    print("Training MCTS+DDQN for a short span to get base weights...")
    trainer = HybridMCTSDDQNTrainer(
        difficulty=1,
        reward_scheme='wave_bonus',
        num_simulations=15,  # Moderate MCTS search
        hidden_dim=128,
        num_res_blocks=2,
        lr=5e-4,
        batch_size=64,
        buffer_size=5000,
        epsilon_decay_steps=2000,
        eval_freq=10,
        eval_episodes=1,
        checkpoint_dir='checkpoints/fast_mcts',
        log_dir='runs/fast_mcts',
        seed=42,
    )
    
    # Train for 50 episodes to let DDQN and DualNet learn *something*
    trainer.train(num_episodes=50, max_steps_per_episode=300)

    print("\nRecording video using MCTS Planner...")
    
    # For recording, we'll use the MCTS search directly to make smart decisions
    engine = PvZEngine(difficulty=1)
    engine.reset(seed=42)
    
    mcts = MCTS(
        network=trainer.dual_net,
        num_simulations=30, # More simulations during evaluation for smarter play
        c_puct=1.0,         # Less exploration
        temperature=0.1     # Greedy choice
    )

    out_path = 'C:\\Users\\Lenovo\\.gemini\\antigravity\\brain\\e55754d7-ea08-4cac-b71c-2394479ccecf\\mcts_smart_run.mp4'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    fps = 15

    for step in range(600):  # Record up to 600 steps
        # Render
        frame = engine.render()
        if writer is None:
            writer = cv2.VideoWriter(out_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

        # MCTS Action Selection
        policy, _ = mcts.search(engine)
        
        # Valid actions only
        mask = SimulatorWrapper.get_action_mask(engine.state)
        policy = policy * mask
        if policy.sum() > 0:
            policy /= policy.sum()
            action = np.argmax(policy)
        else:
            action = 0 # fallback to Wait
            
        # Step
        engine.step(action)
        if engine.state.game_over or engine.state.victory:
            print(f"Game ended at step {step}")
            break

    # flush last frame a few times to pause at the end
    for _ in range(15):
        writer.write(bgr_frame)
        
    writer.release()
    print(f"Video saved to {out_path}")

if __name__ == "__main__":
    main()
