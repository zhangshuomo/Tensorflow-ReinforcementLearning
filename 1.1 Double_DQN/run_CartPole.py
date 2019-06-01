import gym
from DoubleDQN import DoubleDQN

env = gym.make('CartPole-v0')
env = env.unwrapped	# r1, r2
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
agent = DoubleDQN(4,2,epsilon_increment=0.01, memory_space=1000, batch_size=500)
episode = 1000
step = 0

for epi in range(episode):
	observation = env.reset()
	while True:
		env.render()
		action = agent.choose_action(observation)
		observation_, reward, done, info = env.step(action)
		x, x_v, theta, theta_v = observation_
		r1 = (env.x_threshold - abs(x))/ env.x_threshold - 0.8
		r2 = (env.theta_threshold_radians - abs(theta))/ env.theta_threshold_radians - 0.5
		reward = r1 + r2
		agent.add_to_memory(observation, action, reward, observation_)
		if step > 200:	agent.learn()
		step += 1
		if done: break
		observation = observation_
	print('episode %d finished.'%epi)
