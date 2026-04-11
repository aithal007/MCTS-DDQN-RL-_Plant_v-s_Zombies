"""MCTS + DDQN hybrid RL module for PvZ-RL.

Implements an AlphaZero-inspired hybrid that combines:
- Monte Carlo Tree Search (MCTS) with PUCT selection
- Double DQN (DDQN) for value estimation and policy learning
- Self-play training with experience replay
"""

from .mcts import MCTS
from .networks import PvZDualNet
from .hybrid_trainer import HybridMCTSDDQNTrainer

__all__ = ["MCTS", "PvZDualNet", "HybridMCTSDDQNTrainer"]
