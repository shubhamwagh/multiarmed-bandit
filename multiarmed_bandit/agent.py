import numpy as np
from multiarmed_bandit.bandit import MultiArmedBandit
from multiarmed_bandit.policy import Policy

from typing import Optional, Union

class Agent(object):
    """
    An Agent is able to take one of a set of actions at each time step. The
    action is chosen using a strategy based on the history of prior actions
    and outcome observations.
    """

    def __init__(self, bandit: MultiArmedBandit, policy: Policy, prior: float = 0.0, step_size: Optional[float] = None):
        super(Agent, self).__init__()
        self.policy = policy
        self.num_arms = bandit.num_arms
        self.prior = prior
        self.step_size = step_size
        self._value_estimates = prior * np.ones(self.num_arms)
        self.action_attempts = np.zeros(self.num_arms)
        self.t = 0
        self.last_action = None

    def __str__(self) -> str:
        return f'f/{self.policy}'

    def reset(self) -> None:
        """
        Resets the agent's memory to an initial state.
        """
        self._value_estimates[:] = self.prior
        self.action_attempts[:] = 0
        self.last_action = None
        self.t = 0

    def choose(self) -> int:
        action = self.policy.choose(self)
        self.last_action = action
        return action

    def observe(self, reward: Union[int, float]) -> None:
        self.action_attempts[self.last_action] += 1

        if self.step_size is None:
            g = 1 / self.action_attempts[self.last_action]
        else:
            g = self.step_size
        q = self._value_estimates[self.last_action]

        self._value_estimates[self.last_action] += g * (reward - q)
        self.t += 1

    @property
    def value_estimates(self) -> np.ndarray:
        return self._value_estimates
