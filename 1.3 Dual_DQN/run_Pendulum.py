import gym
import numpy as np
from Dual_DQN import Dual_DQN

MEMORY_SPACE = 3000
ACTION_SPACE = 25
env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
step = 0 
agent = Dual_DQN(n_features=3, n_actions=ACTION_SPACE, memory_space=MEMORY_SPACE, epsilon_increment=0.001,dual=True)


observation = env.reset()
while True:
	env.render()
	action = agent.choose_action(observation)
	f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)
	observation_, reward, done, info = env.step(np.array([f_action]))
	reward /= 10
	agent.add_to_memory(observation,action,reward,observation_)
	if step > MEMORY_SPACE:
		agent.learn()
	if step > 30000:
		break
	step += 1
	observation = observation_
