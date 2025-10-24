from typing import Dict, Tuple
from pydantic import BaseModel
from enum import Enum
import numpy as np
import random


class QConfig(BaseModel):
    epsilon: float
    alpha: float
    gamma: float


class Action(Enum):
    COOPERATE = 0
    DEFECT = 1


class Payoffs:
    def __init__(self, B: float, C: float) -> None:
        self.benefit: float = B
        self.cost: float = C
        self.strategy_payoffs()

    def strategy_payoffs(self) -> None:
        self.CC: float = self.benefit - (self.cost / 2)
        self.CD: float = self.benefit - self.cost
        self.DC: float = self.benefit
        self.DD: float = 0


class Player:
    def __init__(self, config: QConfig, payoffs: Payoffs) -> None:
        self.payoffs: Payoffs = payoffs
        self.constants: QConfig = config
        self.qtable: Dict[Action, float] = {Action.COOPERATE: 0.0, Action.DEFECT: 0.0}

    def move(self) -> Action:
        if random.random() < self.constants.epsilon:
            return random.choice(list(self.qtable))
        return max(self.qtable, key=self.qtable.get)

    def update(self, action: Action, reward: float) -> None:
        self.qtable[action] += self.constants.alpha * (
            reward
            + self.constants.gamma * max(self.qtable.values())
            - self.qtable[action]
        )

    def resolve_reward(self, my_action: Action, other_action: Action) -> float:
        mapping: Dict[Tuple[Action, Action], float] = {
            (Action.COOPERATE, Action.COOPERATE): self.payoffs.CC,
            (Action.COOPERATE, Action.DEFECT): self.payoffs.CD,
            (Action.DEFECT, Action.COOPERATE): self.payoffs.DC,
            (Action.DEFECT, Action.DEFECT): self.payoffs.DD,
        }
        return mapping[(my_action, other_action)]