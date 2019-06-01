from DQN import DeepQnetwork
from maze import Maze
import numpy as np

total_episodes = 1000
maze =  Maze()
dqn = DeepQnetwork(2, 4)
step = 0

for episode in range(total_episodes):
	state = maze.reset()
	maze.title('maze-%d'%(episode+1))
	while True:
		action = dqn.choose_action(state)
		next_state, reward, done = maze.step(action)
		dqn.add_to_memory(state,action,reward,next_state)
		# dqn.test(state)
		if step > 200 and step % 5 == 0:
			dqn.learn()
		state = next_state
		if done:
			break
		step += 1
