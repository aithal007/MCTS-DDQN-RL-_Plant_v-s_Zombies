import sys
from pathlib import Path
import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from pvz_rl.mcts.hybrid_trainer import MCTS
from pvz_rl.mcts.networks import PvZDualNet
from pvz_rl.mcts.simulator_wrapper import SimulatorWrapper
from pvz_rl.sim.env import PvZSimEnv

def main():
    print("Initializing MCTS Planner for full lookahead test...")
    # Instantiate untutored network - pure tree search will drive decisions!
    net = PvZDualNet(obs_dim=144, action_dim=316, hidden_dim=128)
    net.eval()
    
    # 100 simulations per tick = deep strategic lookahead!
    mcts = MCTS(
        network=net,
        num_simulations=100, 
        c_puct=1.5,
        temperature=1.0  # safe temperature, we do argmax later anyway
    )

    env = PvZSimEnv(difficulty=1, render_mode="rgb_array", reward_shaping=False)
    env.reset(seed=42)
    engine = env.unwrapped.engine

    out_path = 'C:\\Users\\Lenovo\\.gemini\\antigravity\\brain\\e55754d7-ea08-4cac-b71c-2394479ccecf\\mcts_pure_planner.mp4'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    fps = 15

    print(f"Recording lookahead run to {out_path} ...")
    for step in range(800):
        # Render frame
        frame = env.render()
        if writer is None:
            writer = cv2.VideoWriter(out_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

        # Let the MCTS explore the parallel universes for this step
        policy, root_val = mcts.search(engine)
        
        # Valid actions only
        mask = SimulatorWrapper.get_action_mask(engine.state)
        policy = policy * mask
        if policy.sum() > 0:
            policy /= policy.sum()
            action = np.argmax(policy)
        else:
            action = 0 # Wait

        env.step(action)
        engine = env.unwrapped.engine
        
        if step % 50 == 0:
            print(f"Step {step}/800... [Root MCTS Expected Value: {root_val:.3f}]")
            
        if engine.state.game_over or engine.state.victory:
            print(f"Episode concluded at step {step}")
            break

    # Add padding to video end
    for _ in range(15):
        writer.write(bgr_frame)
    writer.release()
    print("Done generating MCTS video!")

if __name__ == "__main__":
    main()
