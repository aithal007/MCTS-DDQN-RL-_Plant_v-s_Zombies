import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

with open('runs/rl_last/training_log.json', 'r') as f:
    data = json.load(f)

episodes = [d["episode"] for d in data]
rewards = [d["reward"] for d in data]
kills = [d.get("zombies_killed", 0) for d in data]
lengths = [d["length"] for d in data]

# 1. Reward Distribution Histogram Split
plt.figure(figsize=(10, 5))
split_idx = 100
plt.hist(rewards[:split_idx], bins=20, alpha=0.5, label='Episodes 1-100', color='blue')
if len(rewards) > split_idx:
    plt.hist(rewards[split_idx:], bins=20, alpha=0.5, label='Episodes 101+', color='orange')
plt.xlabel("Episode Reward")
plt.ylabel("Frequency")
plt.title("Reward Distribution Shift: Early vs Late Training")
plt.legend()
plt.tight_layout()
plt.savefig("reward_distribution.png", dpi=150)
plt.close()

# 2. Kills vs Length Scatter Plot
plt.figure(figsize=(10, 6))
sc = plt.scatter(lengths, kills, c=episodes, cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.5)
plt.colorbar(sc, label="Episode Number")
plt.xlabel("Survival Length (Steps)")
plt.ylabel("Zombies Killed")
plt.title("Zombies Killed vs. Survival Length")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kills_vs_length.png", dpi=150)
plt.close()

print("Graphs generated successfully.")
