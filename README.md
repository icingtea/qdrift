# QDrift

A Q-learning simulation framework for studying cooperation dynamics in the **Snowdrift Game** (also known as the Hawk-Dove game) using reinforcement learning.

## Overview

QDrift simulates a population of agents playing the Snowdrift Game and learning to cooperate or defect through Q-learning. Each agent uses Q-tables to learn optimal strategies in repeated interactions with other agents, allowing you to observe emergent cooperation patterns over time in this classic game theory scenario.

## Features

- **Snowdrift Game Simulation**: Models the classic game theory scenario where cooperation is costly but mutual defection is worse
- **Q-Learning Agents**: Agents learn through repeated interactions using Q-tables with configurable learning parameters
- **Cooperation Dynamics**: Study how cooperation emerges and evolves in a population facing the snowdrift dilemma
- **Parameter Testing**: Systematically analyze how different parameters affect convergence to cooperation
- **Smoothing Filters**: Apply Gaussian, box (mean), or median filters to visualize trends
- **Configurable Payoffs**: Customize benefit and cost structures for the snowdrift scenario

## Installation

Requires Python 3.13 or higher.

```bash
git clone <repository-url>
cd qdrift
uv sync
```

## Quick Start

### Running a Single Simulation

```bash
# Basic run with 100 agents for 200,000 steps
python main.py --run --n 100

# With Gaussian smoothing
python main.py --run --n 100 --gaussian --sigma 100

# With box filter
python main.py --run --n 100 --box --window_size 1000
```

### Parameter Testing

Test how a parameter affects cooperation convergence:

```bash
# Test epsilon values from 0 to 1 in increments of 0.1
python main.py --n 100 --paramtest epsilon --param_increment 0.1

# Test alpha values and save all run visualizations
python main.py --n 100 --paramtest alpha --param_increment 0.05 --save_runs
```

## Configuration

Edit `config.toml` to set default parameters:

```toml
[simulation]
epsilon = 0.5              # Exploration rate
alpha = 0.01               # Learning rate
gamma = 0                  # Discount factor
alpha_decay_rate = 0.9999  # How quickly learning rate decays
epsilon_decay_rate = 0.9999 # How quickly exploration rate decays

[payoffs]
B = 4  # Benefit of cooperation
C = 2  # Cost of cooperation
```

### Payoff Structure

The simulation implements the **Snowdrift Game** payoff matrix:
- **Both Cooperate (CC)**: B - C/2 (share the cost of cooperation)
- **Cooperate, Other Defects (CD)**: B - C (pay full cost, but still get benefit)
- **Defect, Other Cooperates (DC)**: B (get benefit without paying cost)
- **Both Defect (DD)**: 0 (worst outcome for both)

Unlike the Prisoner's Dilemma, the Snowdrift Game makes mutual defection the worst outcome, creating different evolutionary dynamics where some level of cooperation is typically favored.

## Command Line Arguments

### Required
- `--n`: Number of agents in the simulation

### Mode Selection (choose one)
- `--run`: Run a single simulation
- `--paramtest`: Test a parameter across a range of values

### Parameter Testing
- `--paramtest {epsilon,alpha,gamma,alpha_decay_rate,epsilon_decay_rate}`: Parameter to test
- `--param_increment`: Step size for parameter values (must be < 1)
- `--save_runs`: Save visualizations for all parameter test runs

### Smoothing Filters (optional)
- `--gaussian`: Apply Gaussian smoothing
  - `--sigma`: Standard deviation for Gaussian filter
- `--box`: Apply box (mean) filter
  - `--window_size`: Window size for averaging
- `--median`: Apply median filter
  - `--window_size`: Window size for median

### Other Options
- `--steps`: Number of timesteps (default: 200,000)

## Output

### Single Run
Generates a plot in the `runs/` directory showing:
- Cooperation ratio over time
- Applied smoothing filters (if specified)
- Simulation parameters in the corner

Files are named `run_<hash>.png` based on parameter configuration.

### Parameter Testing
Generates a plot showing:
- Parameter value vs. convergence cooperation ratio
- How different parameter values affect final cooperation levels

Files are named `<parameter>_vs_convergence.png`

## How It Works

1. **Initialization**: N agents are created, each with their own Q-table for cooperation/defection
2. **Interaction**: At each timestep, agents are randomly paired
3. **Learning**: After each interaction, agents update their Q-tables based on:
   - The reward received
   - Their current Q-values
   - Learning rate (alpha) and discount factor (gamma)
4. **Exploration**: Agents balance exploration (random actions) and exploitation (best known action)
5. **Convergence**: Over time, epsilon and alpha decay, and agents converge to stable strategies
