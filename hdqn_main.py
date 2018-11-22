import matplotlib
#%matplotlib inline
#matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt


import torch.optim as optim

from envs.mdp import StochasticMDPEnv
from agents.hdqn_mdp import hDQN, OptimizerSpec
from hdqn import hdqn_learning
from utils.plotting import plot_episode_stats, plot_visited_states
from utils.schedule import LinearSchedule

if __name__ == "__main__":

	plt.style.use('ggplot')

	NUM_EPISODES = 12000
	BATCH_SIZE = 128
	GAMMA = 1.0
	REPLAY_MEMORY_SIZE = 1000000
	LEARNING_RATE = 0.00025
	ALPHA = 0.95
	EPS = 0.01

	optimizer_spec = OptimizerSpec(
	    constructor=optim.RMSprop,
	    kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
	)

	exploration_schedule = LinearSchedule(50000, 0.1, 1)

	agent = hDQN(
	    optimizer_spec=optimizer_spec,
	    replay_memory_size=REPLAY_MEMORY_SIZE,
	    batch_size=BATCH_SIZE,
	)

	env = StochasticMDPEnv()

	agent, stats, visits = hdqn_learning(
	    env=env,
	    agent=agent,
	    num_episodes=NUM_EPISODES,
	    exploration_schedule=exploration_schedule,
	    gamma=GAMMA,
	)


	plot_episode_stats(stats)