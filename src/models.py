import os
import tomllib
from enum import Enum
from typing import Optional

from pydantic import BaseModel
from rich.console import Console


DEFAULT_STEPS = 200000
TOML_PATH = os.path.join("config.toml")

CONSOLE = Console()


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
    gaussian: Optional[bool] = None
    box: Optional[bool] = None
    median: Optional[bool] = None
    window: Optional[int] = None
    sigma: Optional[float] = None


class Payoffs:
    def __init__(self):
        with open(TOML_PATH, "rb") as f:
            payoff_dict = tomllib.load(f).get("payoffs", {})

        self.benefit: float = payoff_dict.get("B")
        self.cost: float = payoff_dict.get("C")

        self.ess = (self.benefit - self.cost) / (self.benefit - (self.cost / 2))

        self.strategy_payoffs()

    def strategy_payoffs(self):
        self.CC = self.benefit - (self.cost / 2)
        self.CD = self.benefit - self.cost
        self.DC = self.benefit
        self.DD = 0.0
