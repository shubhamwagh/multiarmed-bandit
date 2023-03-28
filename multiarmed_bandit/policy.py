import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


class Policy(ABC):
    """
    A policy prescribes an action to be taken based on the memory of an agent
    """

    def __init__(self):
        pass

    def __str__(self):
        return 'generic policy'

    @abstractmethod
    def choose(self, agent):
        raise NotImplementedError('Implement custom choose method for your agent')


class EpsilonGreedyPolicy(Policy):
    """
    The Epsilon-Greedy policy will choose a random action with probability
    epsilon and take the best apparent approach with probability 1-epsilon.
    If multiple actions are tied for best choice, then a random action from that
    subset is selected.
    """

    def __init__(self, epsilon):
        super(EpsilonGreedyPolicy, self).__init__()
        self.epsilon = epsilon

    def __str__(self):
        return '\u03B5-Greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, agent):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.value_estimates))
        else:
            action = np.argmax(agent.value_estimates)
            check = np.where(agent.value_estimates == agent.value_estimates[action])[0]
            if len(check) == 1:
                return action
            else:
                return np.random.choice(check)


class GreedyPolicy(EpsilonGreedyPolicy):
    """
    The Greedy policy only takes the best apparent action, with ties broken by
    random selection. This can be seen as a special case of EpsilonGreedy where
    epsilon = 0 i.e. always exploit.
    """

    def __init__(self):
        super(GreedyPolicy, self).__init__(epsilon=0.0)

    def __str__(self):
        return 'greedy'


class RandomPolicy(EpsilonGreedyPolicy):
    """
    The Random policy randomly selects from all available actions with no
    consideration to which is apparently best. This can be seen as a special
    case of EpsilonGreedy where epsilon = 1 i.e. always explore.
    """

    def __init__(self):
        super(RandomPolicy, self).__init__(epsilon=1.0)

    def __str__(self):
        return 'random'
