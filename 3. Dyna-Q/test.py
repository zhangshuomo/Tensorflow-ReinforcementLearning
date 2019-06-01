from Dyna_Q import Dyna_Q
from maze import Maze 
import matplotlib.pyplot as plt

total_episodes=500
env=Maze()
agent=Dyna_Q(4, 0.3)
steps=[]
for epi in range(total_episodes):
	state=env.reset()
	step=0
	while True:
		action=agent.chooseAction(state)
		next_state, reward, done = env.step(action)
		agent.learn(state, action, reward, next_state)
		agent.saveToModel(state, action, reward, next_state)
		agent.learnFromModel(30)
		if done:
			result = 'Well' if reward == 5 else 'Bad'
			steps.append(step)	
			print('episode:{}\ttotal_steps:{}\tresult:{}'.format(epi+1, step, result))
			break
		state=next_state
		step+=1
plt.plot(steps)
plt.show()	
