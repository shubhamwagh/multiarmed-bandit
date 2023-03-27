import time
import numpy as np
from abc import ABC, abstractmethod
from bandits import BernoulliBandit

from typing import List, 


class Solver(ABC):
    def __init__(self, bandit: BernoulliBandit):
        """
        bandit (Bandit): the target bandit to solve.
        """
        assert isinstance(bandit, BernoulliBandit)
        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = [0] * self.bandit.n_arms
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = [0.]  # History of cumulative regret.

    def update_regret(self, i: int):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_prob - self.bandit.probs_arms[i]
        self.regrets.append(self.regret)

    @abstractmethod
    def q_values(self):
        raise NotImplementedError

    @abstractmethod
    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps: int):
        assert self.bandit is not None
        for _ in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)


class EpsilonGreedy(Solver):
    def __init__(self, bandit: BernoulliBandit, eps: float, init_prob: float = 1.0):
        super(EpsilonGreedy, self).__init__(bandit)

        assert 0.0 <= eps <= 1.0
        self.eps = eps
        self._q_values = [init_prob] * self.bandit.n_arms

    @property
    def q_values(self):
        return self._q_values

    def run_one_step(self) -> int:
        if np.random.random() < self.eps:
            # random exploration
            i = np.random.randint(0, self.bandit.n_arms)
        else:
            # exploitation i.e. pick the best one
            i = np.argmax(self.q_values)

        r = self.bandit.generate_reward(i)
        self.q_values[i] += 1. / (self.counts[i] + 1) * (r - self.q_values[i])
        return i
