# PvZ-RL

Reinforcement learning agents for **Plants vs Zombies** -- featuring a built-in Pygame simulator and support for the original Steam game.

```
┌─────────────────────────────────────────────────────┐
│                   Gymnasium API                      │
│         gym.make("PvZSim-v0")  /  gym.make("PvZSteam-v0")  │
├────────────────────────┬────────────────────────────┤
│   Pygame Simulator     │    Steam OG Wrapper        │
│   (pvz_rl.sim)         │    (pvz_rl.steam)          │
│                        │                            │
│  • 7 plant types       │  • Screen capture (mss)    │
│  • 5 zombie types      │  • Mouse automation        │
│  • Wave-based levels   │  • Auto sun collection     │
│  • ~10k steps/sec      │  • ~6 steps/sec            │
├────────────────────────┴────────────────────────────┤
│               RL Agents (pvz_rl.agents)             │
│      PPO  •  DQN  •  MaskablePPO (sb3-contrib)     │
│   Custom feature extractors  •  Action masking      │
└─────────────────────────────────────────────────────┘
```

## Installation

```bash
cd pvz-rl

# Base install (simulator only)
pip install -e .

# With Steam game support (Windows only)
pip install -e ".[steam]"

# Dev tools
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quick Start

```bash
# Quick demo -- trains PPO for ~2 minutes, then evaluates
python scripts/demo.py

# Train with full config
python scripts/train.py --config configs/train_sim.yaml

# Train with CLI overrides
python scripts/train.py --algo ppo --difficulty 3 --timesteps 200000

# Play the simulator yourself
python scripts/evaluate.py --human --difficulty 2

# Watch a trained agent play
python scripts/evaluate.py --model checkpoints/pvz_rl_final.zip --render

# Evaluate without rendering (prints stats)
python scripts/evaluate.py --model checkpoints/pvz_rl_final.zip --episodes 50

# Train on the real Steam game (game must be running)
python scripts/train.py --config configs/train_steam.yaml
```

## Environments

| | **PvZSim-v0** (Simulator) | **PvZSteam-v0** (Steam OG) |
|---|---|---|
| **Observation** | Structured dict (grid, zombie map, sun, cooldowns) | RGB image (160x120) |
| **Action** | `Discrete(316)`: noop + 7 plants x 5 rows x 9 cols | `Discrete(53)`: noop + 7 seed slots + 45 cells |
| **Speed** | ~10,000 steps/sec | ~6 steps/sec (real-time) |
| **Plants** | Sunflower, Peashooter, Wall-nut, Snow Pea, Cherry Bomb, Repeater, Potato Mine | Whatever's in your seed tray |
| **Zombies** | Normal, Conehead, Buckethead, Flag, Pole Vaulter | Real game zombies |
| **Reward** | Shaped: kills, damage, plant loss, win/loss | External (vision model needed) |
| **Action Masking** | Yes (`get_action_mask()`) | No |
| **Platform** | Any | Windows only |

### Simulator Observation Space

| Key | Shape | Description |
|---|---|---|
| `grid` | `(5, 9)` int8 | Plant type ID at each cell (0 = empty) |
| `grid_hp` | `(5, 9)` float32 | Normalised plant HP [0, 1] |
| `zombie_map` | `(5, 9)` float32 | Total normalised zombie HP per cell |
| `sun` | `(1,)` float32 | Current sun / 2000 |
| `cooldowns` | `(7,)` float32 | Normalised cooldown per plant type |
| `wave_progress` | `(1,)` float32 | Fraction of waves spawned |

After `FlattenDictObs` wrapper: a single float32 vector of length **144**.

### Reward Shaping (Simulator)

| Event | Reward |
|---|---|
| Zombie killed | +2.0 |
| Plant lost | -1.0 |
| Damage dealt | +0.005 per HP |
| Time penalty | -0.001 per tick |
| Victory | +200.0 |
| Game over | -100.0 |

## Training Tips

- **Action masking** dramatically improves training speed. The default `make_sim_env()` enables it. Use `MaskablePPO` from `sb3-contrib` for best results.
- **Curriculum learning** is built into the training script. Difficulty auto-scales during training (configure stages in the YAML config).
- Start with difficulty 1 -- it uses fewer, weaker zombies. Difficulty 10 throws bucketheads and pole vaulters in large waves.
- Monitor training with TensorBoard: `tensorboard --logdir runs/`

## Project Structure

```
pvz-rl/
├── pyproject.toml
├── configs/
│   ├── train_sim.yaml          # Simulator training config
│   └── train_steam.yaml        # Steam game training config
├── scripts/
│   ├── train.py                # Main training entry point
│   ├── evaluate.py             # Eval + human play mode
│   └── demo.py                 # Quick 50k-step demo
└── pvz_rl/
    ├── __init__.py             # Gym env registration
    ├── sim/                    # Pygame PvZ simulator
    │   ├── constants.py        # Plant/zombie stats, grid layout, waves
    │   ├── game.py             # Tick-based game engine
    │   ├── renderer.py         # Pygame rendering (headless or windowed)
    │   └── env.py              # Gymnasium env (dict obs + discrete actions)
    ├── steam/                  # Steam OG game interface
    │   ├── capture.py          # Screen capture, input automation, sun detection
    │   └── env.py              # Gymnasium env (RGB image obs)
    ├── agents/
    │   ├── factory.py          # make_agent() / make_sim_env()
    │   ├── feature_extractors.py  # MLP + CNN extractors for SB3
    │   └── wrappers.py         # FlattenDictObs, ActionMaskWrapper
    └── utils/
        ├── helpers.py
        └── visualization.py    # Training curves, video recording
```

## License

MIT

# MCTS-DDQN-RL-_Plant_v-s_Zombies
