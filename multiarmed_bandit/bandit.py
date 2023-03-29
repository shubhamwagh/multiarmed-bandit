import numpy as np
from abc import ABC, abstractmethod

from typing import Tuple, Optional, List, Union


class MultiArmedBandit(ABC):
    """
    A Multi-armed Bandit
    """

    def __init__(self, num_arms: int) -> None:
        self.num_arms = num_arms
        self.action_values = np.zeros(num_arms)
        self.optimal = 0

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def pull(self, action: int) -> Tuple[Union[int, float], bool]:
        raise NotImplementedError


class GaussianBandit(MultiArmedBandit):
    """
    Gaussian bandits model the reward of a given arm as normal distribution with
    provided mean and standard deviation.
    """

    def __init__(self, num_arms: int, mu: float = 0.0, sigma: float = 1.0):
        super(GaussianBandit, self).__init__(num_arms=num_arms)
        self.mu = mu
        self.sigma = sigma
        self.reset()

    def reset(self) -> None:
        self.action_values = np.random.normal(self.mu, self.sigma, self.num_arms)
        self.optimal = np.argmax(self.action_values)

    def pull(self, action: int) -> Tuple[float, bool]:
        return (np.random.normal(self.action_values[action]),
                action == self.optimal)


class BinomialBandit(MultiArmedBandit):
    """
    The Binomial distribution models the probability of an event occurring with
    p probability k times over N trials i.e. get heads on a p-coin k times on
    N flips.
    In the bandit scenario, this can be used to approximate a discrete user
    rating or "strength" of response to a single event.
    """

    def __init__(self, num_arms: int, num_trials: int, prob_arms: Optional[List[float]] = None,
                 num_samples: Optional[int] = None):
        super(BinomialBandit, self).__init__(num_arms=num_arms)
        assert prob_arms is None or len(prob_arms) == num_arms

        self.num_trials = num_trials
        self.prob_arms = prob_arms
        self.num_samples = num_samples

        self._samples = None
        self._cursor = 0

        self.reset()

    def reset(self) -> None:
        if self.prob_arms is None:
            self.action_values = np.random.uniform(size=self.num_arms)
        else:
            self.action_values = self.prob_arms

        if self.num_samples is not None:
            self._samples = np.random.binomial(self.num_trials, self.action_values,
                                               size=(self.num_samples, self.num_arms))
            self._cursor = 0

        self.optimal = np.argmax(self.action_values)

    def pull(self, action: int) -> Tuple[int, bool]:
        return self.sample[action], action == self.optimal

    @property
    def sample(self) -> np.ndarray:
        if self._samples is None:
            return np.random.binomial(self.num_trials, self.action_values)
        else:
            val = self._samples[self._cursor]
            self._cursor += 1
            return val


class BernoulliBandit(BinomialBandit):
    """
    The Bernoulli distribution models the probability of a single event
    occurring with p probability i.e. get heads on a single p-coin flip. This is
    the special case of the Binomial distribution where N=1.
    In the bandit scenario, this can be used to approximate a hit or miss event,
    such as if a user clicks on a headline, ad, or recommended product.
    """

    def __init__(self, num_arms: int, prob_arms: Optional[List[float]] = None, num_samples: Optional[int] = None):
        super(BernoulliBandit, self).__init__(num_arms=num_arms, num_trials=1, prob_arms=prob_arms,
                                              num_samples=num_samples)
