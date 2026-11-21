# Smart Dice Board Game (RL Course Project - Phase 1)

A clean Python repository implementing the **Smart Dice Board Game** Gymnasium environment, deterministic board configurations, random policy evaluation, and a BFS shortest-path baseline.

> **Phase 1 Focus**: Gymnasium environment design, transition rules, reward modeling, verification tests, and planning baseline. RL training algorithms (Q-Learning, SARSA, DQN, etc.) are reserved for subsequent phases.

---

## 1. Project Overview & Environment Design

The Smart Dice Board Game is modeled as a Markov Decision Process (MDP) on a board of $N$ numbered squares ($1, 2, \dots, N$):

- **Start Position**: Square $1$.
- **Goal Position**: Square $N$. Reaching square $N$ terminates the episode.
- **Action Space**: `gymnasium.spaces.Discrete(6)`
  - Standard 0-indexed actions: $a \in \{0, 1, 2, 3, 4, 5\}$
  - Internally mapped to dice values: $\text{dice\_value} = a + 1$ (values $1$ through $6$)
  - Fully compatible with Stable-Baselines3 and standard RL frameworks.
- **State / Observation Space**: `gymnasium.spaces.Discrete(N + 1)`
  - Current board position $s \in \{1, 2, \dots, N\}$
  - Square 0 is unused. `env.observation_space.contains(obs)` is `True` after `reset()` and on every `step()`.
- **Max-Steps Limit**: Avoids infinite loops by setting `truncated = True` when the step limit is reached.
- **Deterministic Seeded Resets**: `reset(seed=...)` allows deterministic and reproducible rollouts.

### Reward Function

| Event / Outcome | Reward | Description |
| :--- | :---: | :--- |
| **Reach Final Square ($N$)** | `+100` | Reaching square $N$ directly or through a ladder/snake ends the episode (`terminated = True`). |
| **Land on Ladder** | `+20` | Landing on the base of a ladder moves the agent forward to the ladder's top. |
| **Land on Snake** | `-20` | Landing on the head of a snake moves the agent backward to the snake's tail. |
| **Normal Move** | `-1` | Landing on a regular square. Penalizes excessive rolls. |
| **Invalid Move ($s + \text{dice} > N$)** | `-10` | Moving beyond $N$ is invalid: position stays unchanged, episode continues (`terminated = False`). |

### Transition Rules

Given current position $s$ and chosen action $a \in \{0, \dots, 5\}$:
1. Dice value: $\text{dice\_value} = a + 1$.
2. Target square: $s_{target} = s + \text{dice\_value}$.
3. If $s_{target} > N$: invalid move. Agent remains at $s$, reward is $-10$, `invalid_move = True`.
4. If $s_{target} == N$: goal reached. Agent moves to $N$, reward is $+100$, `terminated = True`.
5. If $s_{target}$ is a ladder base: agent climbs to ladder top $L_{end} > s_{target}$. If $L_{end} == N$, reward is $+100$ and `terminated = True`; otherwise reward is $+20$, `ladder_applied = True`.
6. If $s_{target}$ is a snake head: agent slides to snake tail $S_{end} < s_{target}$. If $S_{end} == N$, reward is $+100$ and `terminated = True`; otherwise reward is $-20$, `snake_applied = True`.
7. Otherwise: agent advances to $s_{target}$, reward is $-1$.

---

## 2. Required Board Configurations

All board configurations are defined as configurable data objects (`BoardConfig`) rather than hardcoded logic.

1. **Board 30** (`N=30`, 3 snakes, 3 ladders):
   - Ladders: `3 -> 12`, `8 -> 18`, `15 -> 24`
   - Snakes: `14 -> 4`, `21 -> 9`, `27 -> 7`
2. **Board 50** (`N=50`, 5 snakes, 5 ladders):
   - Ladders: `4 -> 14`, `12 -> 28`, `20 -> 36`, `29 -> 42`, `37 -> 47`
   - Snakes: `17 -> 6`, `25 -> 10`, `34 -> 16`, `43 -> 22`, `48 -> 26`
3. **Board 100** (`N=100`, 8 snakes, 8 ladders):
   - Ladders: `4 -> 25`, `13 -> 39`, `28 -> 55`, `41 -> 62`, `54 -> 76`, `63 -> 84`, `72 -> 91`, `80 -> 96`
   - Snakes: `31 -> 9`, `45 -> 16`, `58 -> 23`, `67 -> 38`, `78 -> 44`, `88 -> 52`, `93 -> 65`, `98 -> 40`

---

## 3. Installation

Ensure Python 3.9+ is installed, then install requirements:

```bash
pip install -r requirements.txt
```

Optionally install the package in editable mode:

```bash
pip install -e .
```

---

## 4. Quickstart: Using the Environment

```python
import gymnasium as gym
from smart_dice import SmartDiceEnv

# Load Board 30
env = SmartDiceEnv(config="30")
obs, info = env.reset(seed=42)
print(f"Starting position: {obs}")  # 1

# Action 1 corresponds to dice value 2 (1 + 2 = 3 -> climbs ladder to 12)
obs, reward, terminated, truncated, info = env.step(1)
print(f"New position: {obs}, Reward: {reward}, Info: {info}")
```

---

## 5. Running Verification Tests

Run the full `pytest` suite:

```bash
pytest -v
```

The test suite validates:
- Observation space containment after `reset()` and on every `step()` (`env.observation_space.contains(obs)`)
- Valid action space is `Discrete(6)` with actions `0..5`
- Action mapping: action 0 means dice 1, action 5 means dice 6
- Ladder and snake transitions landing exactly on $N$ correctly terminate the episode
- Stable-Baselines3 compliance via `check_env` and a smoke test PPO training run
- Normal movement (-1 reward, correct position update)
- Ladder transitions (+20 reward, forward move, `ladder_applied=True`)
- Snake transitions (-20 reward, backward move, `snake_applied=True`)
- Invalid moves beyond $N$ (-10 reward, state unchanged, non-terminating)
- Goal termination at $N$ (+100 reward, `terminated=True`)
- Deterministic seeded resets & reproducibility
- Validation of all three board configurations
- BFS shortest path optimality and execution verification
- Random rollout execution across all three boards

---

## 6. Running the Random-Policy Rollout

Evaluate random rollouts on all three boards:

```bash
python run_random_rollout.py --board all --episodes 50
```

---

## 7. Running the BFS Shortest-Path Baseline

Compute the minimum-roll shortest path and optimal deterministic trajectory on all boards:

```bash
python run_bfs.py --board all
```

---

## 8. Repository Layout

```
.
├── smart_dice/
│   ├── envs/
│   │   ├── board_config.py     # BoardConfig dataclass & preset configurations
│   │   └── smart_dice_env.py   # SmartDiceEnv Gymnasium implementation
│   ├── baselines/
│   │   └── bfs.py              # BFS shortest-path graph solver
│   └── scripts/
│       ├── random_rollout.py   # Random policy evaluation script
│       └── run_bfs.py          # BFS baseline runner
├── tests/
│   ├── test_env.py             # Unit tests for dynamics, rewards, transitions, SB3
│   ├── test_configs.py         # Tests for board sizes, snakes, ladders
│   ├── test_bfs.py             # Tests for BFS solver on all boards
│   └── test_rollout.py         # Tests for random rollout runs
├── run_random_rollout.py       # Top-level rollout CLI
├── run_bfs.py                  # Top-level BFS CLI
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Build & package metadata
└── README.md                   # Documentation and usage guide
```
