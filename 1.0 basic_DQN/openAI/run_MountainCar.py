import gym
from DQN import DeepQnetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

agent = DeepQnetwork(2,3,memory_space=3000,batch_size=32,steps_update=300,epsilon_increment=0.0002)
episode = 500
step = 0

for epi in range(episode):
	observation = env.reset()
	while True:
		env.render()
		action = agent.choose_action(observation)
		observation_, reward, done, info = env.step(action)
		position, velocity = observation_
		reward = abs(position - (-0.5))
		agent.add_to_memory(observation,action,reward,observation_)
		if step > 1000:
			agent.learn()
		step += 1
		observation = observation_
		if done:
			print('episode %d finished.'%epi)
			break

