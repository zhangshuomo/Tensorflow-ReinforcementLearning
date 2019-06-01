import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_OF_STATES=7
ACTIONS=['left','right']
epsilon=0.9
MAX_EPISODES=13
GAMMA=0.9
ALPHA=0.5


def obtain_q_table(n_of_states,actions):
	q_table=pd.DataFrame(np.zeros((n_of_states,len(actions))),columns=actions)
	return q_table

def choose_action(state,q_table):
	row_of_state = q_table.iloc[state,:]
	if np.random.uniform() > epsilon or (row_of_state==0).all():
		action = np.random.choice(q_table.columns)
	else:
		action = row_of_state.idxmax()
	return action

def environment(cur_state,action):
	reward = 0
	if action == 'right':
		if cur_state != N_OF_STATES-2:
			state = cur_state + 1
		else:
			state = 'termination'
			reward = 1
	else:
		if cur_state == 0:	
			state = cur_state
		else:
			state = cur_state - 1
	return state,reward

def show_game_status(cur_state,step,episode,n_of_states):
	status = ['-']*(n_of_states-1) + ['T']
	if cur_state != 'termination':
		status[cur_state] = 'o'
		status = ''.join(status) 
		print('\r%s  the %d step,episode:%d                    '%(status,step,episode+1),end='')
		time.sleep(0.3)
	else:
		print('\repisode %d finished. Totall step:%d           '%(episode+1,step),end='')
		time.sleep(1)
	
def play():
	steps=[]
	q_table=obtain_q_table(N_OF_STATES,ACTIONS)
	for episode in range(MAX_EPISODES):
		cur_state = 0
		step = 0
		show_game_status(cur_state,step,episode,N_OF_STATES)
		while cur_state != 'termination':
			action = choose_action(cur_state,q_table)
			next_state,reward = environment(cur_state,action)
			step += 1
			show_game_status(next_state,step,episode,N_OF_STATES)
			if next_state == 'termination':
				update_target = reward
			else:
				update_target = reward + GAMMA*q_table.iloc[next_state,:].max()
			q_table.ix[cur_state,action] += ALPHA*(update_target - q_table.ix[cur_state,action])
			cur_state = next_state
		steps.append(step)
	return steps,q_table

if __name__ == '__main__':
	steps,q_table = play()
	print('Q table:')
	print(q_table)
	fig,ax = plt.subplots(1,1)
	ax.plot(steps,'k-',lw=5)
	plt.show()
