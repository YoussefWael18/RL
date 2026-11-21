# Smart Dice Board Game (RL Course Project - Phase 1)

A clean Python repository implementing the **Smart Dice Board Game** Gymnasium environment, deterministic board configurations, cycle validation, random policy evaluation, and a BFS shortest-path baseline.

> **Phase 1 Focus**: Gymnasium environment design, transition rules, reward modeling, cycle validation, verification tests, and planning baseline. RL training algorithms (Q-Learning, SARSA, DQN, etc.) are reserved for subsequent phases.

---

## 1. Project Overview & Environment Design

The Smart Dice Board Game is modeled as a Markov Decision Process (MDP) on a board of $N$ numbered squares ($1, 2, \dots, N$):

- **Start Position**: Square $1$.
- **Goal Position**: Square $N$. Reaching square $N$ terminates the episode.
- **Action Space**: `gymnasium.spaces.Discrete(6)`
  - Standard 0-indexed actions: $a \in \{0, 1, 2, 3, 4, 5\}$
  - Internally mapped to dice values: $\text{dice\_value} = a + 1$ (values $1$ through $6$)
  - In `SmartDiceEnv.step()`, `action_space.contains(action)` is verified before int casting. Floats (e.g. `1.0`), strings (`"1"`), bools (`True`), negative values, and values above 5 are strictly rejected with a `ValueError`.
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

## 2. Board Configurations & Positive Reward Cycle Validation

All board configurations are defined as configurable data objects (`BoardConfig`).

### Strict Coordinate Validation
In `BoardConfig`, snake and ladder coordinates, board size $N$, and `max_steps` must be strict integers. Boolean values (`True`, `False`), floats (`4.0`), and numeric strings (`"4"`) are rejected with `TypeError`.

### Positive Reward Cycle Detection
Because ladders grant $+20$ and snakes penalize $-20$, certain snake/ladder arrangements can form infinite cycles with net positive reward ($\sum r > 0$), allowing an RL agent to cycle indefinitely rather than navigate to the goal.
The environment includes `detect_positive_reward_cycle()` using the Bellman-Ford algorithm on the inverted-reward graph ($w = -r$). By default (`allow_positive_reward_cycles=False`), `BoardConfig` raises a `ValueError` if a positive reward cycle is detected.

### Preset Boards
1. **Board 30** (`N=30`, 3 snakes, 3 ladders - Loop-safe):
   - Ladders: `3 -> 12`, `8 -> 18`, `15 -> 24`
   - Snakes: `14 -> 4`, `21 -> 9`, `27 -> 7`
2. **Board 50** (`N=50`, 5 snakes, 5 ladders - Loop-safe):
   - Ladders: `4 -> 14`, `12 -> 28`, `20 -> 36`, `29 -> 42`, `37 -> 47`
   - Snakes: `17 -> 6`, `25 -> 10`, `34 -> 16`, `43 -> 22`, `48 -> 26`
3. **Board 100 (Loop-Safe)** (`N=100`, 8 snakes, 8 ladders):
   - Ladders: `4 -> 25`, `13 -> 39`, `28 -> 55`, `41 -> 62`, `54 -> 76`, `63 -> 84`, `72 -> 91`, `80 -> 96`
   - Snakes: `31 -> 9`, `45 -> 16`, `58 -> 32`, `67 -> 43`, `77 -> 65`, `86 -> 66`, `93 -> 74`, `97 -> 85`
   - Verified 100% loop-safe with no positive reward cycles.
4. **Board 100 (Assignment Exploit Baseline)** (`board100_assignment`):
   - Preserves the original assignment coordinates exhibiting the $+40$ reward cycle:
     `84 -> 52 -> 76 -> 96 -> 40 -> 62 -> 84` (+40 every 6 moves). Available for exploit demonstration and comparative analysis.

---

## 3. Installation

Ensure Python 3.9+ is installed, then install requirements:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 4. Quickstart: Using the Environment

```python
import gymnasium as gym
from smart_dice import SmartDiceEnv

# Load loop-safe Board 30
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
- Positive reward cycle detection and prevention
- Detection of the exact `board100_assignment` $+40$ exploit cycle
- Rejection of non-integer coordinates (bools, floats, strings)
- Rejection of invalid action types (bools, floats, strings, negative, $>5$)
- Observation space containment after `reset()` and on every `step()` (`env.observation_space.contains(obs)`)
- Stable-Baselines3 compliance via `check_env` and a smoke test PPO training run
- Normal movement, ladder transitions, snake transitions
- Invalid moves beyond $N$ (-10 reward, state unchanged, non-terminating)
- Goal termination at $N$ (+100 reward, `terminated=True`)
- Deterministic seeded resets & reproducibility
- BFS shortest path optimality and execution verification
- Random rollout execution across all three boards

---

## 6. Running the Random-Policy Rollout

Evaluate random rollouts on all boards:

```bash
python run_random_rollout.py --board all --episodes 50
```

---

## 7. Running the BFS Shortest-Path Baseline

Compute the minimum-roll shortest path and optimal deterministic trajectory on all boards:

```bash
python run_bfs.py --board all
```
