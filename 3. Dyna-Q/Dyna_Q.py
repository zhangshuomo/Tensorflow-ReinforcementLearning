import pandas as pd
import numpy as np
import random

class Dyna_Q(object):
	def __init__(self, n_actions, alpha, gamma=0.9, epsilon=0.9):
		self.table=pd.DataFrame(columns=list(range(n_actions)), dtype=np.float32)
		self.model=pd.DataFrame(columns=['action','reward','state_'])
		self.alpha=alpha
		self.gamma=gamma
		self.epsilon=epsilon

	def chooseAction(self, state):
		if state not in self.table.index:
			self._addToMemory(state)
		if random.random() > self.epsilon or self.table.loc[state].all() == 0:
			a=random.choice(self.table.columns)
		else:
			a=self.table.loc[state].idxmax()
		return a

	def learn(self, state, action, reward, state_):
#		if state not in self.table.index:
#			self._addToMemory(state)
		if state_ not in self.table.index:
			self._addToMemory(state_)
		objective=reward + self.gamma * self.table.loc[state_].max()
		self.table.loc[state, action] += self.alpha * (objective - self.table.loc[state, action])

	def saveToModel(self, state, action, reward, state_):
		if state not in self.model.index:
			self.model.loc[state]=[action,reward,state_]

	def learnFromModel(self, n_times):
		for i in range(n_times):
			s=random.choice(self.model.index)
			row=self.model.loc[s,:]
			a, r, s_ = row.iloc[0], row.iloc[1], row.iloc[2]
			self.learn(s, a, r, s_)
			
	def _addToMemory(self, state):
		series_table=pd.Series({a:0 for a in self.table.columns},name=state)
		self.table=self.table.append(series_table)

