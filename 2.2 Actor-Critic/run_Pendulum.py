from ActorCritic import ActorCritic
import gym

env = gym.make('Pendulum-v0')
env = env.unwrapped
agent = ActorCritic(env.observation_space.shape[0],continuous=True,action_bounds=[-2,2])
step = 0
total_episode = 200
total_step = 1000

for epi in range(total_episode):
	observation = env.reset()
	step = 0
	total_reward = 0
	while True:
		env.render()
		action = agent.choose_action(observation)
		observation_, reward, done, info = env.step(action)
		reward *= 5
		total_reward += reward
		agent.learn(observation,action,reward,observation_)
		step += 1
		if step >= total_step: 
			print('episode: %d finished, reward:'%(epi+1),int(total_reward/5))		
			break
		observation = observation_
