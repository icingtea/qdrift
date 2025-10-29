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


DEFAULT_STEPS = 200000
TOML_PATH = os.path.join("config.toml")


class Action(Enum):
    COOPERATE = 1
    DEFECT = 0


class Parameter(Enum):
    EPSILON = 0
    ALPHA = 1
    GAMMA = 2
    ALPHA_DECAY_RATE = 3
    EPSILON_DECAY_RATE = 4


class Mode(Enum):
    RUN = 0
    PARAMTEST = 1


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


class Payoffs:
    def __init__(self):
        with open(TOML_PATH, "rb") as f:
            payoff_dict = tomllib.load(f).get("payoffs")

        self.benefit: float = payoff_dict.get("B")
        self.cost: float = payoff_dict.get("C")

        self.strategy_payoffs()

    def strategy_payoffs(self):
        self.CC = self.benefit - (self.cost / 2)
        self.CD = self.benefit - self.cost
        self.DC = self.benefit
        self.DD = 0.0


class Player:
    def __init__(self, config: QConfig, payoffs: Payoffs, index: int):
        self.index = index
        self.payoffs = payoffs
        self.constants = config
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

    def update(self, self_action: Action, other_action: Action):
        mapping: Dict[Tuple[Action, Action], float] = {
            (Action.COOPERATE, Action.COOPERATE): self.payoffs.CC,
            (Action.COOPERATE, Action.DEFECT): self.payoffs.CD,
            (Action.DEFECT, Action.COOPERATE): self.payoffs.DC,
            (Action.DEFECT, Action.DEFECT): self.payoffs.DD,
        }

        reward = mapping[(self_action, other_action)]
        best_future = max(self.qtable.values())

        self.qtable[self_action] += self.constants.alpha * (
            reward + self.constants.gamma * best_future - self.qtable[self_action]
        )


class Sandbox:
    def __init__(
        self,
        n_agents: int,
        payoffs: Payoffs,
        display_args: DisplayConfig,
        gamma: Optional[float] = None,
        epsilon: Optional[float] = None,
        alpha: Optional[float] = None,
        alpha_decay_rate: Optional[float] = None,
        epsilon_decay_rate: Optional[float] = None,
    ):
        with open(TOML_PATH, "rb") as f:
            config_dict = tomllib.load(f).get("simulation")

        self.payoffs = payoffs

        self.config = QConfig(
            epsilon=epsilon if epsilon is not None else config_dict.get("epsilon"),
            alpha=alpha if alpha is not None else config_dict.get("alpha"),
            gamma=gamma if gamma is not None else config_dict.get("gamma"),
            alpha_decay_rate=alpha_decay_rate
            if alpha_decay_rate is not None
            else config_dict.get("alpha_decay_rate"),
            epsilon_decay_rate=epsilon_decay_rate
            if epsilon_decay_rate is not None
            else config_dict.get("epsilon_decay_rate"),
        )
        self.display_args = display_args

        self.n_agents = n_agents
        self.agents: List[Player] = [
            Player(config=self.config.model_copy(), payoffs=payoffs, index=i)
            for i in range(n_agents)
        ]
        self.roles: np.ndarray = np.zeros(shape=n_agents, dtype=int)

        self.box_filtered_ratios: Optional[np.ndarray] = None
        self.gaussian_filtered_ratios: Optional[np.ndarray] = None
        self.median_filtered_ratios: Optional[np.ndarray] = None

        self.cooperator_ratios: List[float] = []
        self.converging_values: List[float] = []

        self.convergence_value: Optional[float] = None

    def step(self):
        index_perm = np.random.permutation(self.n_agents)
        pairings = [
            (index_perm[i], index_perm[i + 1]) for i in range(0, self.n_agents - 1, 2)
        ]

        for index_1, index_2 in pairings:
            player_1 = self.agents[index_1]
            player_2 = self.agents[index_2]

            move_1 = player_1.move(self.roles)
            move_2 = player_2.move(self.roles)

            player_1.update(move_1, move_2)
            player_2.update(move_2, move_1)

    def run(self, n_steps: int):
        for _ in range(1, n_steps + 1):
            self.step()

            counter = Counter(self.roles)
            cooperator_ratio = counter[Action.COOPERATE.value] / self.n_agents

            if self.agents[0].constants.epsilon < 0.01:
                self.converging_values.append(cooperator_ratio)
            self.cooperator_ratios.append(cooperator_ratio)

        self.cooperator_ratios = np.array(self.cooperator_ratios)
        self.converging_values = np.array(self.converging_values)
        self.convergence_value = np.mean(self.converging_values)

        self.filtered_curve()
        self.display_run(n_steps)

    def filtered_curve(self):
        window_size = self.display_args.window

        if self.display_args.box:
            self.box_filtered_ratios = uniform_filter1d(
                self.cooperator_ratios,
                window_size,
                mode="nearest",
            )

        if self.display_args.gaussian:
            self.gaussian_filtered_ratios = gaussian_filter1d(
                self.cooperator_ratios, sigma=self.display_args.sigma
            )

        if self.display_args.median:
            self.median_filtered_ratios = median_filter(
                self.cooperator_ratios, size=window_size
            )

    def param_test(self, parameter: Parameter, n_steps: int, step_size: float):
        convergences = []
        param_range = np.arange(0, 1 + step_size, step_size)

        for parameter_value in param_range:
            kwargs = {
                "n_agents": self.n_agents,
                "payoffs": self.payoffs,
                "display_args": self.display_args,
                parameter.name.lower(): parameter_value,
            }

            sandbox = Sandbox(**kwargs)
            sandbox.run(n_steps)
            convergences.append(sandbox.convergence_value)

        self.display_param_test(parameter, param_range, convergences)

    def display_param_test(self, parameter: Parameter, param_range, convergences):
        plt.figure(figsize=(8, 5))

        plt.plot(param_range, convergences, marker="o", markersize=3, linewidth=1.5)
        plt.title(f"{parameter.name.capitalize()} vs Convergence Value", fontsize=14)
        plt.xlabel(parameter.name.capitalize())
        plt.ylabel("Convergence Value")

        ax = plt.gca()
        x_ticks = np.arange(0, 1.1, 0.1)
        y_ticks = np.arange(0, 1.1, 0.1)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.minorticks_on()

        plt.tight_layout()

        filename = f"{parameter.name.lower()}_vs_convergence.png"
        plt.savefig(filename, dpi=300)
        plt.close()

    def display_run(self, n_steps: int):
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

        ax = plt.gca()

        y_ticks = np.arange(0, 1.1, 0.1)
        ax.set_yticks(y_ticks)

        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.minorticks_on()

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

        plt.close()


def parse_args() -> Tuple[
    DisplayConfig, int, int, Mode, Optional[Parameter], Optional[float]
]:
    parser = argparse.ArgumentParser(description="qdrift plotting")

    parser.add_argument(
        "--run", action="store_true", help="do a single run from your TOML file"
    )
    parser.add_argument(
        "--n",
        type=int,
        help="number of agents",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="number of timesteps",
        default=DEFAULT_STEPS,
    )

    parser.add_argument(
        "--gaussian",
        action="store_true",
        help="apply gaussian smoothing",
    )
    parser.add_argument(
        "--box",
        action="store_true",
        help="apply box (mean) filter",
    )
    parser.add_argument(
        "--median",
        action="store_true",
        help="apply median filter",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        help="sigma value for gaussian filter",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        help="window size for box or median filters",
    )

    parser.add_argument(
        "--paramtest",
        type=lambda s: Parameter[s.upper()],
        choices=list(Parameter),
        help="analyze a parameter's trends",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        help="define the step size for testing.",
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

    if args.run and args.paramtest:
        sys.exit("ERROR: --run and --paramtest are mutually exclusive.")

    if args.step_size:
        if not args.paramtest:
            sys.exit("ERROR: --step_size must be used with --paramtest.")

        if args.step_size >= 1:
            sys.exit("ERROR: --step-size must be < 1.")

    if not args.run and not args.paramtest:
        sys.exit("ERROR: either --run or --paramtest must be specified.")

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
        Mode.RUN if args.run else Mode.PARAMTEST,
        args.paramtest if args.paramtest else None,
        args.step_size if args.step_size else None,
    )


def main():
    display_args, n_agents, n_steps, mode, parameter, step_size = parse_args()
    payoffs = Payoffs()

    if mode == Mode.RUN:
        sandbox = Sandbox(
            n_agents=n_agents,
            display_args=display_args,
            payoffs=payoffs,
        )

        sandbox.run(n_steps=n_steps)
    elif mode == Mode.PARAMTEST:
        sandbox = Sandbox(
            n_agents=n_agents,
            display_args=display_args,
            payoffs=payoffs,
        )

        sandbox.param_test(parameter, n_steps, step_size)


if __name__ == "__main__":
    main()
