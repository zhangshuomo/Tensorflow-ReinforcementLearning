import gym
import numpy as np
from DoubleDQN import DoubleDQN

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SPACE = 3000
ACTION_SPACE = 11

step = 0 
agent = DoubleDQN(n_features=3, n_actions=ACTION_SPACE, memory_space=MEMORY_SPACE, epsilon_increment=0.001)

observation = env.reset()
while True:
	env.render()
	action = agent.choose_action(observation)
	f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)	# bug2
	observation_, reward, done, info = env.step(np.array([f_action]))	# bug1 the order of the ouput params (step function)!
	reward /= 10
	agent.add_to_memory(observation,action,reward,observation_)
	if step > MEMORY_SPACE:
		agent.learn()
	if step > 30000:
		break
	observation = observation_						# bug3
	step += 1
