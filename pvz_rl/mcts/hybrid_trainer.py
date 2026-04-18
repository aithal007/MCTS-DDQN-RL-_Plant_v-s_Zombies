"""Hybrid MCTS + Double DQN trainer for PvZ-RL.

Combines Monte Carlo Tree Search planning with Double DQN learning:

Training Loop (AlphaZero-inspired):
    1. Self-play: Use MCTS (guided by policy-value network) to generate episodes.
       At each step, MCTS produces an improved policy π_MCTS from visit counts.
    2. Store transitions in replay buffer with MCTS-enhanced targets.
    3. Train DDQN online network:
       - Q-learning loss using MCTS value as auxiliary target
       - Policy distillation loss: cross-entropy with MCTS policy
    4. Periodically sync target network (Double DQN).
    5. Evaluate and checkpoint.

The MCTS acts as a "policy improvement operator": it uses the current
network to look ahead and produce better action selections, which then
serve as training signal to improve the network itself.

References:
    Silver et al. (2017) - AlphaGo Zero
    Silver et al. (2018) - AlphaZero
    van Hasselt et al. (2016) - Double DQN
"""

from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pvz_rl.sim.env import PvZSimEnv
from pvz_rl.sim.constants import NUM_PLANT_TYPES, ROWS, COLS
from pvz_rl.agents.wrappers import FlattenDictObs
from pvz_rl.rewards.reward_shaping import RewardShaperWrapper, RewardConfig, RewardScheme

from .networks import PvZDualNet, DuelingDDQN
from .mcts import MCTS
from .simulator_wrapper import SimulatorWrapper
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, Transition


class HybridMCTSDDQNTrainer:
    """Hybrid MCTS + DDQN training pipeline.

    Args:
        difficulty: Game difficulty (1-10).
        reward_scheme: Reward shaping variant to use.
        reward_kwargs: Extra reward config overrides.
        hidden_dim: Hidden dim for networks.
        num_res_blocks: Residual blocks for policy-value net.
        num_simulations: MCTS simulations per move.
        c_puct: MCTS exploration constant.
        temperature: MCTS temperature for action selection.
        temperature_decay: Decay temperature over training.
        lr: Learning rate for optimizer.
        gamma: Discount factor.
        batch_size: Training batch size.
        buffer_size: Replay buffer capacity.
        target_update_freq: Steps between target network syncs.
        epsilon_start: Initial exploration rate.
        epsilon_end: Final exploration rate.
        epsilon_decay_steps: Steps to decay epsilon.
        mcts_policy_weight: Weight for MCTS policy distillation loss.
        eval_freq: Evaluate every N episodes.
        eval_episodes: Number of eval episodes.
        checkpoint_dir: Where to save checkpoints.
        log_dir: TensorBoard log directory.
        seed: Random seed.
        device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        difficulty: int = 1,
        reward_scheme: str = "kill_penalty",
        reward_kwargs: Optional[dict] = None,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
        num_simulations: int = 50,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        temperature_decay: float = 0.995,
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        mcts_policy_weight: float = 1.0,
        eval_freq: int = 50,
        eval_episodes: int = 10,
        checkpoint_dir: str = "checkpoints/mcts_ddqn",
        log_dir: str = "runs/mcts_ddqn",
        seed: int = 42,
        device: str = "cpu",
    ):
        self.difficulty = difficulty
        self.reward_scheme = reward_scheme
        self.reward_kwargs = reward_kwargs or {}
        self.hidden_dim = hidden_dim
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.mcts_policy_weight = mcts_policy_weight
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.seed = seed
        self.device = device

        # Dimensions
        self.obs_dim = 144  # flattened PvZSim observation
        self.action_dim = 1 + NUM_PLANT_TYPES * ROWS * COLS  # 316

        # Networks
        self.policy_value_net = PvZDualNet(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            num_res_blocks=num_res_blocks,
        )

        self.online_q_net = DuelingDDQN(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
        )

        self.target_q_net = DuelingDDQN(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
        )
        self.target_q_net.load_state_dict(self.online_q_net.state_dict())

        # Optimizers
        self.pv_optimizer = optim.Adam(self.policy_value_net.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.online_q_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size, seed=seed)

        # MCTS
        self.mcts = MCTS(
            network=self.policy_value_net,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature=temperature,
        )

        # Logging
        self.total_steps = 0
        self.total_episodes = 0
        self.training_log: list[dict] = []
        self.eval_log: list[dict] = []

        # Create dirs
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _make_env(self, render_mode=None) -> tuple:
        """Create environment with reward shaping."""
        base_env = PvZSimEnv(
            difficulty=self.difficulty,
            render_mode=render_mode,
            reward_shaping=False,
        )
        config = RewardConfig(
            scheme=RewardScheme(self.reward_scheme),
            **self.reward_kwargs,
        )
        env = RewardShaperWrapper(base_env, config)
        flat_env = FlattenDictObs(env)
        return flat_env, base_env

    def _get_epsilon(self) -> float:
        """Compute current epsilon for exploration."""
        progress = min(1.0, self.total_steps / max(self.epsilon_decay_steps, 1))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def train(self, num_episodes: int = 5000, max_steps_per_episode: int = 10000):
        """Main training loop.

        Args:
            num_episodes: Total episodes of self-play training.
            max_steps_per_episode: Maximum steps per episode.
        """
        print("=" * 70)
        print("  HYBRID MCTS + DDQN TRAINING")
        print("=" * 70)
        print(f"  Episodes:       {num_episodes}")
        print(f"  MCTS sims:      {self.num_simulations}")
        print(f"  Reward scheme:  {self.reward_scheme}")
        print(f"  Difficulty:     {self.difficulty}")
        print(f"  Device:         {self.device}")
        print("=" * 70)
        print()

        best_win_rate = 0.0

        for episode in range(num_episodes):
            ep_start = time.time()

            # --- Self-play episode with MCTS ---
            episode_data = self._self_play_episode(max_steps_per_episode)

            # --- Train on replay buffer ---
            train_metrics = {}
            if self.replay_buffer.is_ready:
                train_metrics = self._train_step()

            # --- Target network update ---
            if self.total_steps % self.target_update_freq == 0:
                self.target_q_net.load_state_dict(self.online_q_net.state_dict())

            # --- Decay temperature ---
            self.mcts.temperature = max(0.1, self.mcts.temperature * self.temperature_decay)

            # --- Logging ---
            ep_time = time.time() - ep_start
            self.total_episodes += 1

            log_entry = {
                "episode": self.total_episodes,
                "total_steps": self.total_steps,
                "reward": episode_data["total_reward"],
                "length": episode_data["length"],
                "victory": episode_data["victory"],
                "zombies_killed": episode_data["zombies_killed"],
                "epsilon": self._get_epsilon(),
                "temperature": self.mcts.temperature,
                "time": ep_time,
                **train_metrics,
            }
            self.training_log.append(log_entry)

            # Print progress
            if (episode + 1) % 1 == 0:
                recent = self.training_log[-10:]
                avg_reward = np.mean([e["reward"] for e in recent])
                avg_length = np.mean([e["length"] for e in recent])
                win_rate = np.mean([e["victory"] for e in recent])
                avg_kills = np.mean([e["zombies_killed"] for e in recent])

                print(
                    f"Ep {self.total_episodes:>5d} | "
                    f"Steps {self.total_steps:>8d} | "
                    f"R {avg_reward:>8.1f} | "
                    f"Len {avg_length:>6.0f} | "
                    f"WR {win_rate:>5.1%} | "
                    f"Kills {avg_kills:>5.1f} | "
                    f"eps {self._get_epsilon():.3f} | "
                    f"tau {self.mcts.temperature:.3f} | "
                    f"{ep_time:.1f}s"
                )

            # --- Evaluation ---
            if (episode + 1) % self.eval_freq == 0:
                eval_result = self._evaluate()
                self.eval_log.append(eval_result)

                print(f"\n  [EVAL] Win rate: {eval_result['win_rate']:.1%} | "
                      f"Avg reward: {eval_result['avg_reward']:.1f} | "
                      f"Avg kills: {eval_result['avg_kills']:.1f}\n")

                if eval_result["win_rate"] > best_win_rate:
                    best_win_rate = eval_result["win_rate"]
                    self._save_checkpoint("best")

            # --- Periodic checkpoint ---
            # --- Periodic checkpoint ---
            self._save_checkpoint(f"ep{self.total_episodes}")
            self._save_logs()

        # Save final
        self._save_checkpoint("final")
        self._save_logs()
        print(f"\nTraining complete! Best win rate: {best_win_rate:.1%}")

    def _self_play_episode(self, max_steps: int) -> dict:
        """Run one self-play episode using MCTS.

        Returns:
            Dict with episode statistics.
        """
        env, base_env = self._make_env()
        obs, info = env.reset(seed=self.seed + self.total_episodes)

        total_reward = 0.0
        transitions = []
        done = False
        step = 0

        while not done and step < max_steps:
            action_mask = base_env.get_action_mask()

            # Use MCTS for action selection (with some epsilon exploration)
            epsilon = self._get_epsilon()
            use_mcts = np.random.random() > epsilon * 0.5  # Mostly use MCTS

            if use_mcts and self.num_simulations > 0:
                # MCTS search
                mcts_policy, mcts_value = self.mcts.search(base_env.engine)

                # Sample action from MCTS policy
                if self.mcts.temperature < 0.1:
                    action = int(np.argmax(mcts_policy))
                else:
                    action = int(np.random.choice(len(mcts_policy), p=mcts_policy))
            else:
                # Epsilon-greedy using Q-network
                action = self.online_q_net.predict_action(obs, action_mask, epsilon)
                mcts_policy = None
                mcts_value = None

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Store transition
            transition = Transition(
                obs=obs.copy(),
                action=action,
                reward=reward,
                next_obs=next_obs.copy(),
                done=done,
                action_mask=action_mask.copy(),
                mcts_policy=mcts_policy,
                mcts_value=mcts_value,
            )
            transitions.append(transition)

            obs = next_obs
            step += 1
            self.total_steps += 1

        # Push to replay buffer
        self.replay_buffer.push_many(transitions)

        env.close()

        return {
            "total_reward": total_reward,
            "length": step,
            "victory": info.get("victory", False),
            "zombies_killed": info.get("zombies_killed", 0),
        }

    def _train_step(self) -> dict:
        """Perform one training step on a batch from the replay buffer.

        Updates both the Q-network (DDQN) and the policy-value network.

        Returns:
            Dict with loss metrics.
        """
        batch = self.replay_buffer.sample_numpy(self.batch_size)

        obs_t = torch.FloatTensor(batch["obs"])
        actions_t = torch.LongTensor(batch["actions"])
        rewards_t = torch.FloatTensor(batch["rewards"])
        next_obs_t = torch.FloatTensor(batch["next_obs"])
        dones_t = torch.FloatTensor(batch["dones"])
        masks_t = torch.BoolTensor(batch["action_masks"])

        weights = torch.ones(self.batch_size)
        if "weights" in batch:
            weights = torch.FloatTensor(batch["weights"])

        # ===== DDQN Loss =====
        # Current Q-values for taken actions
        current_q = self.online_q_net(obs_t, masks_t)
        current_q_values = current_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: use online net to select actions, target net to evaluate
        with torch.no_grad():
            next_q_online = self.online_q_net(next_obs_t)
            next_actions = next_q_online.argmax(dim=1)
            next_q_target = self.target_q_net(next_obs_t)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Bellman target
            target_q = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        # Weighted Huber loss
        td_errors = target_q - current_q_values
        q_loss = (weights * F.smooth_l1_loss(current_q_values, target_q, reduction="none")).mean()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_q_net.parameters(), max_norm=10.0)
        self.q_optimizer.step()

        # Update priorities
        if "indices" in batch:
            self.replay_buffer.update_priorities(
                batch["indices"], td_errors.detach().numpy()
            )

        # ===== Policy-Value Network Loss =====
        pv_loss_total = torch.tensor(0.0)
        policy_loss_val = 0.0
        value_loss_val = 0.0

        if "mcts_policies" in batch:
            mcts_policies_t = torch.FloatTensor(batch["mcts_policies"])
            logits, values = self.policy_value_net(obs_t, masks_t)

            # Policy loss: cross-entropy with MCTS policy
            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(mcts_policies_t * log_probs).sum(dim=1).mean()

            # Value loss: Huber (Smooth L1) with DDQN target (uses DDQN's better value estimates)
            with torch.no_grad():
                value_targets = target_q.unsqueeze(1)
                # Normalize to [-1, 1]
                value_targets = torch.clamp(value_targets / 200.0, -1.0, 1.0)

            value_loss = F.smooth_l1_loss(values, value_targets)

            pv_loss_total = (
                self.mcts_policy_weight * policy_loss + value_loss
            )

            self.pv_optimizer.zero_grad()
            pv_loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), max_norm=10.0)
            self.pv_optimizer.step()

            policy_loss_val = policy_loss.item()
            value_loss_val = value_loss.item()

        return {
            "q_loss": q_loss.item(),
            "policy_loss": policy_loss_val,
            "value_loss": value_loss_val,
        }

    def _evaluate(self) -> dict:
        """Evaluate the agent without MCTS (pure network).

        Returns:
            Dict with evaluation metrics.
        """
        env, base_env = self._make_env()

        rewards = []
        victories = 0
        total_kills = 0

        for ep in range(self.eval_episodes):
            obs, info = env.reset(seed=self.seed + 100000 + ep)
            total_reward = 0.0
            done = False

            while not done:
                mask = base_env.get_action_mask()
                action = self.online_q_net.predict_action(obs, mask, epsilon=0.0)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

            rewards.append(total_reward)
            if info.get("victory", False):
                victories += 1
            total_kills += info.get("zombies_killed", 0)

        env.close()

        return {
            "episode": self.total_episodes,
            "total_steps": self.total_steps,
            "win_rate": victories / self.eval_episodes,
            "avg_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "avg_kills": total_kills / self.eval_episodes,
        }

    def _save_checkpoint(self, name: str):
        """Save model checkpoints."""
        path = Path(self.checkpoint_dir)

        torch.save(
            self.online_q_net.state_dict(),
            path / f"ddqn_online_{name}.pt",
        )
        torch.save(
            self.target_q_net.state_dict(),
            path / f"ddqn_target_{name}.pt",
        )
        torch.save(
            self.policy_value_net.state_dict(),
            path / f"policy_value_{name}.pt",
        )

        # Save optimizer states
        torch.save(
            {
                "q_optimizer": self.q_optimizer.state_dict(),
                "pv_optimizer": self.pv_optimizer.state_dict(),
                "total_steps": self.total_steps,
                "total_episodes": self.total_episodes,
                "temperature": self.mcts.temperature,
            },
            path / f"training_state_{name}.pt",
        )

    def _save_logs(self):
        """Save training logs to JSON."""
        path = Path(self.log_dir)

        with open(path / "training_log.json", "w") as f:
            json.dump(self.training_log, f, indent=2, default=str)

        with open(path / "eval_log.json", "w") as f:
            json.dump(self.eval_log, f, indent=2, default=str)

    def load_checkpoint(self, name: str):
        """Load model from checkpoint."""
        path = Path(self.checkpoint_dir)

        self.online_q_net.load_state_dict(
            torch.load(path / f"ddqn_online_{name}.pt", map_location="cpu")
        )
        self.target_q_net.load_state_dict(
            torch.load(path / f"ddqn_target_{name}.pt", map_location="cpu")
        )
        self.policy_value_net.load_state_dict(
            torch.load(path / f"policy_value_{name}.pt", map_location="cpu")
        )

        state = torch.load(path / f"training_state_{name}.pt", map_location="cpu")
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.pv_optimizer.load_state_dict(state["pv_optimizer"])
        self.total_steps = state["total_steps"]
        self.total_episodes = state["total_episodes"]
        self.mcts.temperature = state["temperature"]
