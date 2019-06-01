import gym
from DQNExperienceReplay import DQNExperienceReplay
import time

env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

agent = DQNExperienceReplay(2,3,memory_space=10000,batch_size=32,steps_update=300,epsilon_increment=0.00005,experience_replay=False)
episode = 500
step = 0

for epi in range(episode):
	observation = env.reset()
	while True:
		env.render()
		action = agent.choose_action(observation)
		observation_, reward, done, info = env.step(action)
		if done: reward = 10
		agent.add_to_memory(observation,action,reward,observation_)
		if step > 10000:	# verify that the memory has been fully filled.
			agent.learn()
		step += 1
		observation = observation_
		if done:
			print('episode %d finished. time:'%epi,time.ctime())
			break

