import cv2
import torch
import numpy as np
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from pvz_rl.mcts.networks import DuelingDDQN
from pvz_rl.sim.env import PvZSimEnv
from pvz_rl.rewards.reward_shaping import RewardShaperWrapper, RewardConfig, RewardScheme
from pvz_rl.agents.wrappers import FlattenDictObs

def main():
    print("Loading environment...")
    base_env = PvZSimEnv(
        difficulty=1,
        render_mode="rgb_array",
        reward_shaping=False,
    )
    config = RewardConfig(scheme=RewardScheme("wave_bonus"))
    env = RewardShaperWrapper(base_env, config)
    flat_env = FlattenDictObs(env)

    print("Loading agent (Online DDQN)...")
    net = DuelingDDQN(obs_dim=144, action_dim=316, hidden_dim=64)
    # The trainer was tested with hidden_dim=64 in the tiny test
    
    ckpt_path = Path("checkpoints/test_mcts/ddqn_online_final.pt")
    if not ckpt_path.exists():
        print(f"Error: Could not find {ckpt_path}")
        return

    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    net.eval()

    output_path = "C:\\Users\\Lenovo\\.gemini\\antigravity\\brain\\e55754d7-ea08-4cac-b71c-2394479ccecf\\mcts_test_run.mp4"
    print(f"Recording video to {output_path}...")
    
    writer = None
    fps = 10

    max_steps = 300
    obs, info = flat_env.reset(seed=42)
    
    for step in range(max_steps):
        # Render frame
        frame = flat_env.render()
        if frame is not None:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)

        # Get action
        mask = base_env.get_action_mask()
        action = net.predict_action(obs, mask, epsilon=0.0)

        # Step
        obs, reward, terminated, truncated, info = flat_env.step(action)
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    if writer is not None:
        writer.release()
    
    flat_env.close()
    base_env.close()
    print("Done!")

if __name__ == "__main__":
    main()
