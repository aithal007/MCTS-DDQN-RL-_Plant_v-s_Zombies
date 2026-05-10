# Reinforcement Learning for Tower Defense: A Hybrid MCTS + DDQN Approach to Plants vs. Zombies

![PvZ RL Banner](https://img.shields.io/badge/Status-Complete-success)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Compatible-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)

Welcome to the **PvZ-RL** repository. This project implements a novel Deep Reinforcement Learning pipeline designed to master the highly complex, sequential decision-making environment of **Plants vs. Zombies**. 

To overcome the immense action space and delayed sparse rewards inherent to Tower Defense games, we developed a **Hybrid Monte Carlo Tree Search (MCTS) + Dueling Double DQN (DDQN)** architecture, trained on a custom, high-speed Python simulator.

---

## 🌟 Key Features

1. **Custom Gymnasium-Compatible Simulator**
   Built from the ground up to allow deterministic, discrete-time MDP execution. The engine is optimized for speed, processing $\sim$10,000 game ticks per second on CPU to enable massive MCTS lookahead rollouts without relying on the physical game executable.
2. **Hybrid MCTS + DDQN Architecture**
   A bidirectional learning pipeline where MCTS acts as a lookahead planner to teach a Prioritized Experience Replay buffer, while a Dueling DDQN grounds the value estimates to teach the MCTS the actual TD-learned rewards.
3. **Advanced Reward Shaping**
   Tower Defense games suffer from extreme sparse rewards (you only know if you survived or died after hundreds of actions). We implemented and analyzed 5 different reward schemes, achieving optimal convergence using a custom **Wave Bonus** formulation with dense micro-action signals.
4. **Action Masking**
   In a grid of 45 cells with 7 plant types, 99% of randomly chosen actions are invalid (e.g., planting without enough sun or placing on an occupied tile). We built dynamic action masking directly into the policy heads to ensure mathematical exploration stability.

---

## 🧠 Algorithmic Architecture

The agent is driven by two neural networks working in tandem:

### 1. PvZDualNet (The Planner)
An AlphaZero-inspired architecture taking the 144-dimensional flattened game state and passing it through a 4-block Residual MLP backbone. It splits into two heads:
* **Policy Head:** Outputs prior probabilities $P(s,a)$ for 316 discrete actions to guide the MCTS PUCT equation.
* **Value Head:** Outputs a scalar $V(s) \in [-1, 1]$ to evaluate leaf nodes without running full terminal rollouts.

### 2. DuelingDDQN (The Actor)
A Dueling network architecture that splits the Q-value estimation into a **Value Stream** and an **Advantage Stream**, allowing the agent to recognize inherently safe states (like an empty board) without needing to compute the exact utility of every single action.

---

## 📊 Experimental Results

Across a 205-episode training run ($\sim$280,000 environment steps), the hybrid agent demonstrated genuine strategic behavior, including:
* **Saving sun** to prioritize high-damage Shooters over basic Wall-nuts.
* **Lane prioritization**, specifically placing plants in rows actively occupied by approaching zombies.
* **Significant performance gains:** A **+28.5% improvement** in average zombie kills per episode over the course of training, paired with an exponentially decaying policy loss.

---

## 🚀 Installation & Usage

### Requirements
* Python 3.10 or higher
* PyTorch
* Gymnasium
* NumPy, Matplotlib

### Setup
Clone the repository and install the local package:
```bash
git clone https://github.com/aithal007/MCTS-DDQN-RL-_Plant_v-s_Zombies.git
cd MCTS-DDQN-RL-_Plant_v-s_Zombies
pip install -e .
```

### Running the Training Pipeline
To launch a hybrid MCTS+DDQN training run, use the provided script. Hyperparameters can be configured directly in `configs/train_mcts.yaml`.

```bash
python scripts/train_mcts.py --config configs/train_mcts.yaml
```

*Note: Due to the heavy reliance on MCTS (50 simulations per step), training is highly CPU-bound. The current pipeline takes approximately 45 seconds per episode on modern multi-core CPUs.*

---

## 📄 Repository Structure
* `pvz_rl/sim/` - The custom determinisitic PvZ Python game engine and Gymnasium wrappers.
* `pvz_rl/mcts/` - The core RL algorithms, including `networks.py` (DualNet/DDQN), `mcts.py` (Tree Search), and `hybrid_trainer.py`.
* `pvz_rl/rewards/` - Gym wrappers for the 5 ablation reward schemes (Sparse, Kill Penalty, Wave Bonus, Time Penalty, Potential Shaping).
* `scripts/` - Executable entry points for training and plotting.
* `configs/` - YAML hyperparameter configurations.
* `runs/` - JSON training logs and PyTorch network checkpoints.

---

## 🏆 Research Report
For an in-depth mathematical breakdown of the PUCT search algorithms, the Dueling DDQN loss functions, and a thorough analysis of 8 distinct "Reward Hacking" bugs encountered during development, please refer to the included **IEEE-formatted research report**.
