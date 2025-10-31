import random
from typing import Dict, Tuple

import numpy as np


from src.models import (
    QConfig,
    Payoffs,
    Action,
)


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
            action = max(self.qtable, key=self.qtable.get)

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
