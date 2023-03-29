import numpy as np
from multiarmed_bandit.bandit import MultiArmedBandit
from multiarmed_bandit.agent import Agent

from typing import List, Tuple


class Environment:
    def __init__(self, bandit: MultiArmedBandit, agents: List[Agent], label: str = "Multi-Armed Bandit"):
        self.bandit = bandit
        self.agents = agents
        self.label = label

    def reset(self) -> None:
        self.bandit.reset()
        for agent in self.agents:
            agent.reset()

    def run(self, trials: int = 100, experiments: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        scores = np.zeros((trials, len(self.agents)))
        optimal = np.zeros_like(scores)

        for _ in range(experiments):
            self.reset()
            for t in range(trials):
                for i, agent in enumerate(self.agents):
                    action = agent.choose()
                    reward, is_optimal = self.bandit.pull(action)
                    agent.observe(reward)

                    scores[t, i] += reward
                    if is_optimal:
                        optimal[t, i] += 1

        return scores / experiments, optimal / experiments
