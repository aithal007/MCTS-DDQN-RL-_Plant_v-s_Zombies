import sys
from pathlib import Path
import cv2

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from sb3_contrib import MaskablePPO
from pvz_rl.rewards.reward_shaping import make_shaped_env

import os
def main():
    model_path = 'checkpoints/maskable_ppo_wave_bonus_d10_s42/best/best_model.zip'
    if not os.path.exists(model_path):
        model_path = 'checkpoints/maskable_ppo_wave_bonus_d10_s42/maskable_ppo_wave_bonus_d10_s42_final.zip'
    # Save the output to the AI artifacts folder to view it directly in chat.
    out_dir = 'C:\\Users\\Lenovo\\.gemini\\antigravity\\brain\\a576d9eb-64d1-4e9d-a746-f48fefe5076b\\artifacts'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'trained_maskable_ppo_d10_horde.mp4')

    print(f"Loading {model_path}...")
    env = make_shaped_env(difficulty=10, render_mode="rgb_array", use_action_mask=True, scheme="wave_bonus")
    model = MaskablePPO.load(model_path, env=env)

    writer = None
    fps = 15
    obs, _ = env.reset(seed=42)
    done = False
    
    print("Recording episode...")
    step = 0
    while not done and step < 4000:  # Allow more loops for long diff 10 waves
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        frame = env.render()
        if frame is not None:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        step += 1

    if writer:
        writer.release()
    print(f"Video saved to {out_path}")
    env.close()

if __name__ == "__main__":
    main()
