import sys
import multiarmed_bandit as mab


def main():
    num_arms = 10
    n_trials = 1000
    n_experiments = 500

    bandit = mab.GaussianBandit(num_arms=num_arms)

    agents = [mab.Agent(bandit, policy=mab.RandomPolicy()),
              mab.Agent(bandit, policy=mab.GreedyPolicy()),
              mab.Agent(bandit, policy=mab.EpsilonGreedyPolicy(epsilon=0.01)),
              mab.Agent(bandit, policy=mab.EpsilonGreedyPolicy(epsilon=0.1))
            ]
    
    env = mab.Environment(bandit, agents, 'Epsilon-Greedy')
    scores, optimal = env.run(n_trials, n_experiments)
    env.plot_results(scores, optimal)


if __name__ == "__main__":
    sys.exit(main())
