import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load training log
with open('runs/rl_last/training_log.json') as f:
    log = json.load(f)

episodes = [e['episode'] for e in log]
rewards   = [e['reward'] for e in log]
kills     = [e['zombies_killed'] for e in log]
q_loss    = [e.get('q_loss', 0) for e in log]
ep_len    = [e['length'] for e in log]
epsilon   = [e['epsilon'] for e in log]

def rolling(arr, w=10):
    return np.convolve(arr, np.ones(w)/w, mode='valid')

ep_roll = episodes[9:]
r_roll  = rolling(rewards)
k_roll  = rolling(kills)

# ─── Fig 1: 4-panel training dynamics ───
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('MCTS + DDQN Training Dynamics — PvZ-RL (131 episodes, ~177k steps)',
             fontsize=13, fontweight='bold', y=0.98)
plt.subplots_adjust(hspace=0.42, wspace=0.35)

ax = axes[0, 0]
ax.plot(episodes, rewards, alpha=0.25, color='steelblue', linewidth=0.8, label='Raw')
ax.plot(ep_roll, r_roll,  color='royalblue', linewidth=2.0, label='Rolling avg (10 ep)')
ax.set_xlabel('Episode'); ax.set_ylabel('Episode Reward')
ax.set_title('(a) Shaped Reward per Episode'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(episodes, kills, alpha=0.25, color='tomato', linewidth=0.8, label='Raw')
ax.plot(ep_roll, k_roll, color='firebrick', linewidth=2.0, label='Rolling avg (10 ep)')
ax.set_xlabel('Episode'); ax.set_ylabel('Zombies Killed')
ax.set_title('(b) Zombies Killed per Episode'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.semilogy(episodes, q_loss, color='darkorange', linewidth=1.2, alpha=0.8)
ax.set_xlabel('Episode'); ax.set_ylabel('Huber Q-Loss (log scale)')
ax.set_title('(c) DDQN Q-Loss Convergence'); ax.grid(alpha=0.3, which='both')

ax = axes[1, 1]
ax.plot(episodes, epsilon, color='mediumseagreen', linewidth=2.0)
ax.set_xlabel('Episode'); ax.set_ylabel('Epsilon')
ax.set_title('(d) Exploration Rate Decay'); ax.grid(alpha=0.3)

plt.savefig('training_curves.png', dpi=160, bbox_inches='tight')
print('Saved training_curves.png')
plt.close()

# ─── Fig 2: Reward distribution across phases ───
fig2, ax2 = plt.subplots(figsize=(9, 5))
phase1 = rewards[:40]
phase2 = rewards[40:90]
phase3 = rewards[90:]
colors = ['#4e79a7', '#f28e2b', '#59a14f']
labels = ['Phase 1 (ep 1-40)', 'Phase 2 (ep 41-90)', 'Phase 3 (ep 91-131)']
parts = ax2.violinplot([phase1, phase2, phase3], positions=[1, 2, 3], showmeans=True)
for pc, c in zip(parts['bodies'], colors):
    pc.set_facecolor(c); pc.set_alpha(0.75)
ax2.set_xticks([1, 2, 3]); ax2.set_xticklabels(labels)
ax2.set_ylabel('Episode Reward')
ax2.set_title('Fig 2: Reward Distribution Across Training Phases')
ax2.grid(alpha=0.3, axis='y')
means = [np.mean(phase1), np.mean(phase2), np.mean(phase3)]
for pos, m in zip([1, 2, 3], means):
    ax2.text(pos, m + 120, f'u={m:.0f}', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('reward_distribution.png', dpi=160, bbox_inches='tight')
print('Saved reward_distribution.png')
plt.close()

# ─── Fig 3: Kills vs Episode length scatter ───
fig3, ax3 = plt.subplots(figsize=(8, 5))
sc = ax3.scatter(ep_len, kills, c=episodes, cmap='viridis', alpha=0.7, s=30)
plt.colorbar(sc, ax=ax3, label='Episode Number')
ax3.set_xlabel('Episode Length (steps)'); ax3.set_ylabel('Zombies Killed')
ax3.set_title('Fig 3: Zombies Killed vs Episode Length (by training progression)')
ax3.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('kills_vs_length.png', dpi=160, bbox_inches='tight')
print('Saved kills_vs_length.png')
plt.close()

# Print stats
total_steps = log[-1]['total_steps']
print()
print('=== KEY STATS ===')
print('Total episodes:', len(log))
print('Total steps:', total_steps)
print('Avg reward first 10:', round(np.mean(rewards[:10]), 1))
print('Avg reward last 10:', round(np.mean(rewards[-10:]), 1))
print('Avg kills first 10:', round(np.mean(kills[:10]), 1))
print('Avg kills last 10:', round(np.mean(kills[-10:]), 1))
print('Max reward:', round(max(rewards), 1), 'at ep', episodes[rewards.index(max(rewards))])
print('Max kills:', max(kills), 'at ep', episodes[kills.index(max(kills))])
print('Avg episode length:', round(np.mean(ep_len), 0))
print('Q-loss ep1:', round(q_loss[0], 4))
print('Q-loss last:', round(q_loss[-1], 4))
