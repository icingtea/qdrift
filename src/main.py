import sys
import random
import tomllib
import argparse
from enum import Enum
from collections import Counter
from typing import Dict, Tuple, List, Optional
import os
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter


class QConfig(BaseModel):
    epsilon: float
    alpha: float
    gamma: float
    alpha_decay_rate: float
    epsilon_decay_rate: float


class DisplayConfig(BaseModel):
    gaussian: Optional[bool]
    box: Optional[bool]
    median: Optional[bool]
    window: Optional[int]
    sigma: Optional[float]


class Action(Enum):
    COOPERATE = 1
    DEFECT = 0


class Payoffs:
    def __init__(self) -> None:
        with open("config.toml", "rb") as f:
            payoff_dict: Dict[str, float] = tomllib.load(f).get("payoffs")

        self.benefit: float = payoff_dict.get("B")
        self.cost: float = payoff_dict.get("C")

        self.strategy_payoffs()

    def strategy_payoffs(self) -> None:
        self.CC: float = self.benefit - (self.cost / 2)
        self.CD: float = self.benefit - self.cost
        self.DC: float = self.benefit
        self.DD: float = 0.0


class Player:
    def __init__(self, config: QConfig, payoffs: Payoffs, index: int) -> None:
        self.index: int = index
        self.payoffs: Payoffs = payoffs
        self.constants: QConfig = config
        self.qtable: Dict[Action, float] = {
            Action.COOPERATE: 0.0,
            Action.DEFECT: 0.0,
        }

    def move(self, role_array: np.ndarray) -> Action:
        if random.random() < self.constants.epsilon:
            action = random.choice(list(self.qtable))
        else:
            max_q = max(self.qtable.values())
            best_actions = [a for a, q in self.qtable.items() if q == max_q]
            action = random.choice(best_actions)

        self.constants.alpha *= self.constants.alpha_decay_rate
        self.constants.epsilon *= self.constants.epsilon_decay_rate

        role_array[self.index] = action.value
        return action

    def update(self, self_action: Action, other_action: Action) -> None:
        mapping: Dict[Tuple[Action, Action], float] = {
            (Action.COOPERATE, Action.COOPERATE): self.payoffs.CC,
            (Action.COOPERATE, Action.DEFECT): self.payoffs.CD,
            (Action.DEFECT, Action.COOPERATE): self.payoffs.DC,
            (Action.DEFECT, Action.DEFECT): self.payoffs.DD,
        }

        reward: float = mapping[(self_action, other_action)]
        best_future: float = max(self.qtable.values())

        self.qtable[self_action] += self.constants.alpha * (
            reward + self.constants.gamma * best_future - self.qtable[self_action]
        )


class Sandbox:
    def __init__(
        self, n_agents: int, payoffs: Payoffs, display_args: DisplayConfig
    ) -> None:
        with open("config.toml", "rb") as f:
            config_dict: Dict[str, float] = tomllib.load(f).get("simulation")

        self.payoffs: Payoffs = payoffs

        self.config: QConfig = QConfig(
            epsilon=config_dict.get("epsilon"),
            alpha=config_dict.get("alpha"),
            gamma=config_dict.get("gamma"),
            alpha_decay_rate=config_dict.get("alpha_decay_rate"),
            epsilon_decay_rate=config_dict.get("epsilon_decay_rate"),
        )
        self.display_args = display_args

        self.n_agents: int = n_agents
        self.agents: List[Player] = [
            Player(config=self.config.model_copy(), payoffs=payoffs, index=i)
            for i in range(n_agents)
        ]
        self.roles: np.ndarray = np.zeros(shape=n_agents, dtype=int)

        self.box_filtered_ratios: Optional[np.ndarray] = None
        self.gaussian_filtered_ratios: Optional[np.ndarray] = None
        self.median_filtered_ratios: Optional[np.ndarray] = None

        self.cooperator_ratios: np.ndarray | List[float] = []
        self.converging_values: np.ndarray | List[float] = []

        self.convergence_value: Optional[float] = None

    def step(self) -> None:
        index_perm: np.ndarray = np.random.permutation(self.n_agents)
        pairings: List[Tuple[int, int]] = [
            (index_perm[i], index_perm[i + 1]) for i in range(0, self.n_agents - 1, 2)
        ]

        for index_1, index_2 in pairings:
            player_1: Player = self.agents[index_1]
            player_2: Player = self.agents[index_2]

            move_1: Action = player_1.move(self.roles)
            move_2: Action = player_2.move(self.roles)

            player_1.update(move_1, move_2)
            player_2.update(move_2, move_1)

    def run(self, n_steps: int) -> None:
        for _ in range(1, n_steps + 1):
            self.step()

            counter: Counter[int] = Counter(self.roles)

            cooperator_ratio = counter[Action.COOPERATE.value] / self.n_agents

            if self.agents[0].constants.epsilon < 0.01:
                self.converging_values.append(cooperator_ratio)
            self.cooperator_ratios.append(cooperator_ratio)

        self.cooperator_ratios = np.array(self.cooperator_ratios)
        self.converging_values = np.array(self.converging_values)
        self.convergence_value = np.mean(self.converging_values)

        self.filtered_curve()

        plt.plot(
            self.cooperator_ratios,
            color="blue",
            alpha=0.7,
            label="Cooperation Ratio",
            lw=3.0,
        )

        if self.box_filtered_ratios is not None:
            plt.plot(
                self.box_filtered_ratios,
                color="darkblue",
                label="Box Smoothed Cooperation Ratio Curve",
                lw=0.5,
            )

        if self.gaussian_filtered_ratios is not None:
            plt.plot(
                self.gaussian_filtered_ratios,
                color="yellow",
                label="Gaussian Smoothed Cooperation Ratio Curve",
                lw=0.5,
            )

        if self.median_filtered_ratios is not None:
            plt.plot(
                self.median_filtered_ratios,
                color="purple",
                label="Median Smoothed Cooperation Ratio Curve",
                lw=0.5,
            )

        plt.xlabel("step")
        plt.ylabel("Ratio")
        plt.title("Cooperator Ratio Over Time")
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(visible=True)

        os.makedirs("runs", exist_ok=True)

        run_params = f"{self.n_agents}_{n_steps}_{self.config.epsilon}_{self.config.alpha}_{self.config.gamma}_{self.config.alpha_decay_rate}_{self.config.epsilon_decay_rate}"
        run_hash = hashlib.md5(run_params.encode()).hexdigest()[:8]
        filename = os.path.join("runs", f"run_{run_hash}.png")

        param_text = (
            f"n={self.n_agents}, steps={n_steps}\n"
            f"ε={self.config.epsilon:.3f}, α={self.config.alpha:.3f}, γ={self.config.gamma:.3f}\n"
            f"α-decay={self.config.alpha_decay_rate:.5f}, ε-decay={self.config.epsilon_decay_rate:.5f}"
        )

        plt.gcf().text(
            0.02,
            0.98,
            param_text,
            fontsize=6,
            va="top",
            ha="left",
            bbox=dict(
                facecolor="white", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.3"
            ),
        )

        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"saved plot to {filename}")

        plt.show()
        plt.close()

    def filtered_curve(self) -> None:
        window_size = self.display_args.window

        if self.display_args.box:
            self.box_filtered_ratios: np.ndarray = uniform_filter1d(
                self.cooperator_ratios,
                window_size,
                mode="nearest",
            )

        if self.display_args.gaussian:
            self.gaussian_filtered_ratios: np.ndarray = gaussian_filter1d(
                self.cooperator_ratios, sigma=self.display_args.sigma
            )

        if self.display_args.median:
            self.median_filtered_ratios: np.ndarray = median_filter(
                self.cooperator_ratios, size=window_size
            )


def parse_args() -> Tuple[DisplayConfig, int, int]:
    parser = argparse.ArgumentParser(description="qdrift plotting")

    parser.add_argument("--n", type=int, help="number of agents")
    parser.add_argument("--steps", type=int, help="number of timesteps", default=200000)
    parser.add_argument(
        "--gaussian", action="store_true", help="apply gaussian smoothing"
    )
    parser.add_argument("--box", action="store_true", help="apply box (mean) filter")
    parser.add_argument("--median", action="store_true", help="apply median filter")

    parser.add_argument("--sigma", type=float, help="sigma value for Gaussian filter")
    parser.add_argument(
        "--window-size", type=int, help="window size for box or median filters"
    )

    args = parser.parse_args()

    if not args.n:
        sys.exit("ERROR: --n is required.")

    if args.box or args.median:
        if args.window_size is None:
            sys.exit("ERROR: --window-size is required when using --box or --median.")

    if args.gaussian:
        if args.sigma is None:
            sys.exit("ERROR: --sigma is required when using --gaussian.")

    if args.window_size and not (args.box or args.median):
        sys.exit("ERROR: --window-size should only be used with --box or --median.")

    if args.sigma and not args.gaussian:
        sys.exit("ERROR: --sigma should only be used with --gaussian.")

    return (
        DisplayConfig(
            box=args.box,
            median=args.median,
            gaussian=args.gaussian,
            window=args.window_size,
            sigma=args.sigma,
        ),
        args.n,
        args.steps,
    )


if __name__ == "__main__":
    display_args, n_agents, n_steps = parse_args()
    payoffs = Payoffs()
    sandbox = Sandbox(n_agents, payoffs, display_args)
    sandbox.run(n_steps)

    print(sandbox.convergence_value)
