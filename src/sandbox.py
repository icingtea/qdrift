import os
import hashlib
import tomllib
import multiprocessing
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track
from scipy.ndimage import uniform_filter1d, gaussian_filter1d, median_filter


from src.models import (
    Payoffs,
    DisplayConfig,
    QConfig,
    Parameter,
    TOML_PATH,
    CONSOLE,
)


from src.agent import Player


class Sandbox:
    _config_cache = None

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
        if Sandbox._config_cache is None:
            with open(TOML_PATH, "rb") as f:
                config_dict = tomllib.load(f).get("simulation", {})

            # fmt: off

            self.config = QConfig(
                epsilon=(
                    epsilon 
                    if epsilon is not None 
                    else config_dict.get("epsilon", 0.1)
                ),

                alpha=(
                    alpha 
                    if alpha is not None 
                    else config_dict.get("alpha", 0.1)
                ),

                gamma=(
                    gamma 
                    if gamma is not None 
                    else config_dict.get("gamma", 0.9)
                ),

                alpha_decay_rate=(
                    alpha_decay_rate
                    if alpha_decay_rate is not None
                    else config_dict.get("alpha_decay_rate", 0.9999)
                ),
                
                epsilon_decay_rate=(
                    epsilon_decay_rate
                    if epsilon_decay_rate is not None
                    else config_dict.get("epsilon_decay_rate", 0.9999)
                )
            )

            # fmt: on

            Sandbox._config_cache = self.config
        else:
            self.config = Sandbox._config_cache.model_copy(deep=True)

        self.payoffs = payoffs
        self.display_args = display_args

        self.n_agents = n_agents
        self.agents: List[Player] = [
            Player(config=self.config.model_copy(deep=True), payoffs=payoffs, index=i)
            for i in range(n_agents)
        ]
        self.roles: np.ndarray = np.zeros(shape=n_agents, dtype=int)

        self.box_filtered_ratios: Optional[np.ndarray] = None
        self.gaussian_filtered_ratios: Optional[np.ndarray] = None
        self.median_filtered_ratios: Optional[np.ndarray] = None

        self.cooperator_ratios: Optional[np.ndarray] = None
        self.converging_values: Optional[np.ndarray] = None

        self.convergence_value: Optional[float] = None

    def step(self):
        index_perm = np.random.permutation(self.n_agents)

        for i in range(0, self.n_agents - 1, 2):
            i1, i2 = index_perm[i], index_perm[i + 1]
            player_1, player_2 = self.agents[i1], self.agents[i2]

            move_1 = player_1.move(self.roles)
            move_2 = player_2.move(self.roles)

            player_1.update(move_1, move_2)
            player_2.update(move_2, move_1)

    def run(
        self,
        n_steps: int,
        save_runs: bool = True,
        progress_bar: bool = True,
    ):
        self.cooperator_ratios = np.zeros(n_steps)
        self.converging_values = np.full(n_steps, np.nan)

        for i in track(range(1, n_steps + 1), disable=not progress_bar):
            self.step()

            cooperator_ratio = float(self.roles.mean())

            self.cooperator_ratios[i - 1] = cooperator_ratio
            if self.agents[0].constants.epsilon < 0.01:
                self.converging_values[i - 1] = cooperator_ratio

        self.convergence_value = np.nanmean(self.converging_values)

        self.filtered_curve()

        if save_runs:
            self.display_run(n_steps)

    def filtered_curve(self):
        window_size = self.display_args.window

        if self.display_args.box and window_size is not None:
            self.box_filtered_ratios = uniform_filter1d(
                self.cooperator_ratios,
                window_size,
                mode="nearest",
            )

        if self.display_args.gaussian and self.display_args.sigma is not None:
            self.gaussian_filtered_ratios = gaussian_filter1d(
                self.cooperator_ratios, sigma=self.display_args.sigma
            )

        if self.display_args.median and window_size is not None:
            self.median_filtered_ratios = median_filter(
                self.cooperator_ratios, size=window_size
            )

    def param_test(
        self,
        param_type: Parameter,
        n_steps: int,
        param_increment: float,
        save_runs: bool,
    ):
        param_range = np.arange(0, 1 + param_increment, param_increment)

        arglist = [
            (param_value, param_type.name.lower(), n_steps, save_runs)
            for param_value in param_range
        ]

        with CONSOLE.status("[#A0E8E3]this will take a while!", spinner="shark"):
            with multiprocessing.Pool() as pool:
                async_result = pool.starmap_async(self._sandbox_worker, arglist)
                results = async_result.get()

        convergences = np.array(results)
        self.display_param_test(param_type, param_range, convergences)

    def _sandbox_worker(
        self,
        param_value: float,
        param_name: str,
        n_steps: int,
        save_runs: bool,
    ) -> float:
        kwargs: Dict[str, Any] = {
            "n_agents": self.n_agents,
            "payoffs": self.payoffs,
            "display_args": self.display_args,
            param_name: param_value,
        }

        sandbox = Sandbox(**kwargs)
        sandbox.run(n_steps, save_runs, False)

        return (
            sandbox.convergence_value if sandbox.convergence_value is not None else 0.0
        )

    def display_param_test(
        self,
        param_type: Parameter,
        param_range: np.ndarray,
        convergences: np.ndarray,
    ):
        plt.figure(figsize=(8, 5))

        plt.plot(param_range, convergences, marker="o", markersize=3, linewidth=1.5)
        plt.title(f"{param_type.name.capitalize()} vs Convergence Value", fontsize=14)
        plt.xlabel(param_type.name.capitalize())
        plt.ylabel("Convergence Value")

        ax = plt.gca()

        x_ticks = np.arange(0, 1.1, 0.1)
        y_ticks = np.arange(0, 1.1, 0.1)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.minorticks_on()

        plt.tight_layout()

        filename = f"{param_type.name.lower()}_vs_convergence.png"
        plt.savefig(filename, dpi=300)
        plt.close()

    def display_run(
        self,
        n_steps: int,
    ):
        plt.figure(figsize=(10, 6))

        if self.cooperator_ratios is not None:
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

        run_id = hashlib.sha1(
            f"{self.n_agents}{n_steps}{self.config.epsilon}{self.config.alpha}{self.config.gamma}{self.config.alpha_decay_rate}{self.config.epsilon_decay_rate}".encode()
        ).hexdigest()[:8]
        filename = os.path.join("runs", f"run_{run_id}.png")

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
        plt.close()
