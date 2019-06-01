import gym
import time
from Policy_Gradient import PolicyGradient

env = gym.make('MountainCar-v0')
env = env.unwrapped
agent = PolicyGradient(2,3,learning_rate=0.02,discount_factor=0.995)
episode = 100

for epi in range(episode):
	observation = env.reset()
	while True:
		action = agent.choose_action(observation)
		observation_, reward, done, info = env.step(action)
		env.render()
		agent.add_to_memory(observation, action, reward)
		if done:
			agent.learn()
			print('episode %d finished, time:'%epi, time.ctime())
			break
		observation = observation_
