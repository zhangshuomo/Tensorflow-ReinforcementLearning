from ActorCritic import ActorCritic
import gym

env = gym.make('CartPole-v0')
env = env.unwrapped
agent = ActorCritic(env.observation_space.shape[0],env.action_space.n)
episode = 1000

for epi in range(episode):
	observation = env.reset()
	total_reward = 0
	while True:
		env.render()
		action = agent.choose_action(observation)
		observation_, reward, done, info = env.step(action)
		if done:
			reward = -20
		agent.learn(observation,action,reward,observation_)
		total_reward += reward
		if done:
			print('episode: %d finished.reward:'%(epi+1), total_reward)
			break
		observation = observation_
