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
        difficulty=6,
        render_mode="rgb_array",
        reward_shaping=False,
    )
    config = RewardConfig(scheme=RewardScheme("wave_bonus"))
    env = RewardShaperWrapper(base_env, config)
    flat_env = FlattenDictObs(env)

    print("Loading agent (Online DDQN)...")
    net = DuelingDDQN(obs_dim=144, action_dim=316, hidden_dim=256)
    
    ckpt_path = Path("checkpoints/mcts_ddqn/ddqn_online_ep17.pt")
    if not ckpt_path.exists():
        print(f"Error: Could not find {ckpt_path}")
        return

    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    net.eval()

    output_path = "C:\\Users\\Lenovo\\.gemini\\antigravity\\brain\\963577ad-cced-4b78-a917-48683d226c25\\artifacts\\mcts_ddqn_ep17_diff6.mp4"
    print(f"Recording video to {output_path}...")
    
    writer = None
    fps = 30 # Let's speed up playback so the user doesn't wait as long to watch it

    max_steps = 3000
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
