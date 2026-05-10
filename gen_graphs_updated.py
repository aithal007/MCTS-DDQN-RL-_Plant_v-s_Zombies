import json
import matplotlib.pyplot as plt
import numpy as np

with open('runs/rl_last/training_log.json', 'r') as f:
    data = json.load(f)

episodes = [d["episode"] for d in data]
rewards = [d["reward"] for d in data]
kills = [d.get("zombies_killed", 0) for d in data]
lengths = [d["length"] for d in data]
q_losses = [d.get("q_loss", 0) for d in data]
epsilons = [d.get("epsilon", 0) for d in data]

# 1. Training Curves (4 subplots: Reward, Kills, Q-Loss, Epsilon)
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Reward
axs[0].plot(episodes, rewards, alpha=0.3, color="blue")
window = min(10, len(rewards)//5)
if len(rewards) >= window:
    smoothed_rew = np.convolve(rewards, np.ones(window)/window, mode="valid")
    axs[0].plot(episodes[window-1:], smoothed_rew, color="blue", linewidth=2, label=f"Moving Avg (n={window})")
axs[0].set_ylabel("Episode Reward")
axs[0].set_title("Training Dynamics over 193 Episodes")
axs[0].legend()

# Kills
axs[1].plot(episodes, kills, alpha=0.3, color="green")
if len(kills) >= window:
    smoothed_kills = np.convolve(kills, np.ones(window)/window, mode="valid")
    axs[1].plot(episodes[window-1:], smoothed_kills, color="green", linewidth=2, label=f"Moving Avg (n={window})")
axs[1].set_ylabel("Zombies Killed")
axs[1].legend()

# Q-Loss
axs[2].plot(episodes, q_losses, color="red", alpha=0.8)
axs[2].set_ylabel("Q-Loss")
axs[2].set_yscale("log")

# Epsilon
axs[3].plot(episodes, epsilons, color="purple", linewidth=2)
axs[3].set_ylabel("Epsilon")
axs[3].set_xlabel("Episode")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()

# 2. Reward Distribution (Violin or Histogram)
plt.figure(figsize=(8, 5))
plt.hist(rewards, bins=20, color='skyblue', edgecolor='black')
plt.xlabel("Episode Reward")
plt.ylabel("Frequency")
plt.title("Distribution of Episode Rewards")
plt.savefig("reward_distribution.png", dpi=150)
plt.close()

# 3. Kills vs Length
plt.figure(figsize=(8, 5))
sc = plt.scatter(lengths, kills, c=episodes, cmap='viridis', alpha=0.7)
plt.colorbar(sc, label="Episode Number")
plt.xlabel("Survival Length (Steps)")
plt.ylabel("Zombies Killed")
plt.title("Zombies Killed vs. Survival Length")
plt.grid(True, alpha=0.3)
plt.savefig("kills_vs_length.png", dpi=150)
plt.close()

print("Graphs generated successfully.")
