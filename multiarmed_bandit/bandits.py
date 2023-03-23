import time
import numpy as np
from abc import ABC, abstractmethod

from typing import Optional, List


class Bandit(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_reward(self, id):
        raise NotImplementedError


class BernoulliBandit(Bandit):
    def __init__(self, n: int, probs: Optional[List[int]] = None):
        super(BernoulliBandit, self).__init__()
        assert probs is None or len(probs) == n
        self.n = n
        if probs is None:
            np.random.seed(int(time.time()))
            self.probs = [np.random.random() for _ in range(self.n)]
        else:
            self.probs = probs
        
        self.best_prob = max(self.probs)
    
    def generate_reward(self, i):
        # player selects i-th machine
        if np.random.random() < self.probs[i]:
            return 1
        return 0
