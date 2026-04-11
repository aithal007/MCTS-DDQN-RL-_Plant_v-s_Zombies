"""Reward shaping module for PvZ-RL.

Provides configurable reward function wrappers implementing:
- Sparse (win/loss only)
- Kill/Penalty (stepwise kills and losses)
- Wave Bonus (intermediate milestones)
- Time Penalty (discourages stalling)
- Potential-Based Shaping (theoretically optimal-preserving)
"""

from .reward_shaping import RewardShaperWrapper, make_shaped_env

__all__ = ["RewardShaperWrapper", "make_shaped_env"]
