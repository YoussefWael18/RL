# Smart Dice Board Game (RL Course Project)

A clean Python repository implementing the **Smart Dice Board Game** Gymnasium environment, deterministic board configurations, BFS shortest-path baseline, and a complete suite of **Tabular and Deep Reinforcement Learning algorithms** (Q-Learning, SARSA, DQN, PPO, A2C).

---

## 1. Project Overview & Environment Design

The Smart Dice Board Game is modeled as a Markov Decision Process (MDP) on a board of $N$ numbered squares ($1, 2, \dots, N$):

- **Start Position**: Square $1$.
- **Goal Position**: Square $N$. Reaching square $N$ terminates the episode.
- **Action Space**: `gymnasium.spaces.Discrete(6)`
  - Standard 0-indexed actions: $a \in \{0, 1, 2, 3, 4, 5\}$
  - Internally mapped to dice values: $\text{dice\_value} = a + 1$ (values $1$ through $6$)
  - In `SmartDiceEnv.step()`, `action_space.contains(action)` is verified before int casting. Floats (e.g. `1.0`), strings (`"1"`), bools (`True`), negative values, and values $\ge 6$ are strictly rejected with a `ValueError`.
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

---

## 2. Board Configurations & Loop-Safety Validation

All board configurations are defined as configurable data objects (`BoardConfig`).

### Strict Coordinate Validation
In `BoardConfig`, snake and ladder coordinates, board size $N$, and `max_steps` must be strict integers. Boolean values (`True`, `False`), floats (`4.0`), and numeric strings (`"4"`) are rejected with `TypeError`.

### Positive Reward Cycle Detection
Because ladders grant $+20$ and snakes penalize $-20$, certain snake/ladder arrangements can form infinite cycles with net positive reward ($\sum r > 0$). The environment includes `detect_positive_reward_cycle()` using the Bellman-Ford algorithm on the inverted-reward graph ($w = -r$). By default (`allow_positive_reward_cycles=False`), `BoardConfig` raises a `ValueError` if a positive reward cycle is detected.

### Preset Boards
1. **Board 30** (`N=30`, 3 snakes, 3 ladders - Loop-safe)
2. **Board 50** (`N=50`, 5 snakes, 5 ladders - Loop-safe)
3. **Board 100 (Loop-Safe)** (`N=100`, 8 snakes, 8 ladders)
4. **Board 100 (Assignment Exploit Baseline)** (`board100_assignment`): Preserves the assignment layout exhibiting the $+40$ exploit cycle (`84 -> 52 -> 76 -> 96 -> 40 -> 62 -> 84`).

---

## 3. Phase 2: Kaggle-Ready RL Training

The primary training driver is located at [`notebooks/phase2_training.ipynb`](notebooks/phase2_training.ipynb).

### Notebook Highlights
- **Kaggle setup.** Upload the notebook and run all cells. It includes the Smart Dice package source and installs missing Python dependencies from PyPI. Enable Kaggle Internet access if the kernel needs to install dependencies.
- **Execution modes.** The notebook defaults to full training with `SMOKE_TEST = False`. Set it to `True` for a short end-to-end check.
- **Five Algorithms Trained & Compared**:
  1. **Tabular Q-Learning**: Off-policy TD control learning optimal action values.
  2. **Tabular SARSA**: On-policy TD control learning cautious policies around snakes.
  3. **Deep Q-Network (DQN)**: Deep value-based learning with experience replay.
  4. **Proximal Policy Optimization (PPO)**: On-policy actor-critic with clipped surrogate loss.
  5. **Advantage Actor-Critic (A2C)**: Synchronous actor-critic baseline.
- **BFS baseline.** Finds the minimum number of moves for each board. Every five training episodes, a greedy policy checkpoint is evaluated against the BFS move count. Convergence requires a five-checkpoint rolling mean within the larger of one move or 10% of that count for `CONV_WINDOW` consecutive checkpoints.
- **Artifacts.** On Kaggle the notebook saves the following files under `/kaggle/working/artifacts`. Local runs use `./artifacts`.
  - Model checkpoints (`.npy` for tabular, `.zip` for SB3 models)
  - `results_summary.csv`
  - Plots:
    - `total_reward_vs_episodes.png`
    - `avg_moves_vs_episodes.png`
    - `success_rate_vs_episodes.png`
    - `q_learning_vs_sarsa.png`
    - `dqn_vs_ppo_vs_a2c.png`

---

## 4. Historical Smoke-Test Results

These numbers are from an earlier smoke test and are for pipeline verification only. The original convergence values used a previous criterion and have been omitted. Run the notebook in its default full-training mode to generate the benchmark table and plots.

| Board | Algorithm | Avg Moves | Success Rate (%) | Avg Cumulative Reward |
| :--- | :--- | :---: | :---: | :---: |
| **Board 30** | **BFS (Optimal Baseline)** | **3.0** | **100.0%** | **+140.0** |
| Board 30 | Tabular Q-Learning | 3.00 | 100.0% | +140.00 |
| Board 30 | Tabular SARSA | 3.00 | 100.0% | +140.00 |
| Board 30 | SB3 DQN | 5.00 | 100.0% | +138.00 |
| Board 30 | SB3 PPO | 3.00 | 100.0% | +140.00 |
| Board 30 | SB3 A2C | 5.00 | 100.0% | +138.00 |
| **Board 50** | **BFS (Optimal Baseline)** | **4.0** | **100.0%** | **+160.0** |
| Board 50 | Tabular Q-Learning | 4.00 | 100.0% | +160.00 |
| Board 50 | Tabular SARSA | 4.00 | 100.0% | +160.00 |
| Board 50 | SB3 DQN | 6.00 | 100.0% | +158.00 |
| Board 50 | SB3 PPO | 4.00 | 100.0% | +160.00 |
| Board 50 | SB3 A2C | 11.00 | 100.0% | +111.00 |
| **Board 100** | **BFS (Optimal Baseline)** | **7.0** | **100.0%** | **+157.0** |
| Board 100 | Tabular Q-Learning | 7.00 | 100.0% | +157.00 |
| Board 100 | Tabular SARSA | 7.00 | 100.0% | +157.00 |
| Board 100 | SB3 DQN | 10.00 | 100.0% | +154.00 |
| Board 100 | SB3 PPO | 10.00 | 100.0% | +154.00 |
| Board 100 | SB3 A2C | 19.00 | 100.0% | +145.00 |

---

## 5. Running Tests

Run the full verification test suite:

```bash
pytest -v
# 35 passed
```
