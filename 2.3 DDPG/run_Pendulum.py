from DDPG import DDPG
import gym
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v0')
env = env.unwrapped
agent = DDPG(env.observation_space.shape[0], 1, 2)
episodes = 200
max_step = 200
rewards = []
flag = True
for epi in range(episodes):
	observation = env.reset()
	step = 0
	total_reward = 0
	while step < max_step:
		action = agent.choose_action(observation)
		observation_, reward, done, info = env.step(action)
		reward /= 10
		total_reward += reward
		agent.add_to_memory(observation, action, reward, observation_)
		if epi * max_step + step > 10000:
			if flag:
				print('begin')
				flag = False
			agent.learn()
			env.render()
		observation = observation_
		step += 1
	rewards.append(total_reward)
	print('episode %d finished. reward:'%(epi+1), int(total_reward))

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(rewards,lw=3,c='green')
plt.show()
