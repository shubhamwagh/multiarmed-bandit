import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bandit import MultiArmedBandit
from agent import Agent

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

    def plot_results(self, scores: np.ndarray, optimal: np.ndarray):
        sns.set_style('white')
        sns.set_context('talk')
        plt.subplot(2, 1, 1)
        plt.title(self.label)
        plt.plot(scores)
        plt.ylabel('Average Reward')
        plt.legend(self.agents, loc=4)
        plt.subplot(2, 1, 2)
        plt.plot(optimal * 100)
        plt.ylim(0, 100)
        plt.ylabel('% Optimal Action')
        plt.xlabel('Time Step')
        plt.legend(self.agents, loc=4)
        sns.despine()
        plt.show()
