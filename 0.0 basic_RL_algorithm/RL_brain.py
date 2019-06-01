import pandas as pd
import numpy as np

class RL(object):
    def __init__(self, actions, learning_rate=0.5, discount_factor=0.9, epsilon=0.9):
        self.actions = actions.copy()
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float32)
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def chooseAction(self, state):
        x = np.random.uniform()
        if state not in self.q_table.index:  # #
            self.add_state(state)
        # print(self.q_table)
        row = self.q_table.loc[state]  # return a pd.Series
        if x > self.epsilon or all(row == 0):  # #
            action = np.random.choice(self.actions)
        else:
            action = np.random.choice(row[row == row.max()].index) # #
        return action

    def add_state(self, state):
        s = pd.Series(np.zeros((1, len(self.actions))).squeeze(), index=self.actions, dtype='float32', name=state)  ##
        self.q_table = self.q_table.append(s)

    def learn(*args): # a strange overriding method.
        pass


class QLearning(RL):
    def __init__(self, actions, learning_rate=0.5, discount_factor=0.9, epsilon=0.9):
        super(QLearning,self).__init__(actions,learning_rate,discount_factor,epsilon)

    def learn(self, state, action, reward, next_state):
        if next_state not in self.q_table.index:
            self.add_state(next_state)
        if next_state == 'terminition':
            update_target = reward
        else:
            update_target = reward + self.gamma * self.q_table.loc[next_state].max()
        self.q_table.loc[state, action] += self.alpha * (update_target - self.q_table.loc[state, action])

class Sarsa(RL):
    def __init__(self, actions, learning_rate=0.5, discount_factor=0.9, epsilon=0.9):
        super(Sarsa,self).__init__(actions,learning_rate,discount_factor,epsilon)

    def learn(self, state, action, reward, next_state, next_action):
        if next_state == 'termination':
            update_target = reward
        else:
            update_target = reward + self.gamma * self.q_table.loc[next_state,next_action]
        self.q_table.loc[state, action] += self.alpha * (update_target - self.q_table.loc[state, action])

class SarsaLambda(RL):
    def __init__(self, actions, learning_rate=0.5, discount_factor=0.9, epsilon=0.9, trace_factor=0.5):
        super(SarsaLambda,self).__init__(actions,learning_rate,discount_factor,epsilon)
        self.trace_factor = trace_factor
        self.trace_table = pd.DataFrame(columns=self.actions,dtype="float32")

    def add_state(self, state):
        s = pd.Series(np.zeros((1, len(self.actions))).squeeze(), index=self.actions, dtype='float32', name=state)
        self.q_table = self.q_table.append(s)
        self.trace_table = self.trace_table.append(s)

    def learn(self, state, action, reward, next_state, next_action):
        if next_state == 'terminition':
            update_target = reward
        else:
            update_target = reward + self.gamma * self.q_table.loc[next_state, next_action]
      # method 1
      # self.trace_table[state,action] += 1
      # method 2
        self.trace_table.loc[state,:] *= 0
        self.trace_table.loc[state,action] = 1
        self.q_table += self.alpha * (update_target - self.q_table.loc[state, action]) * self.trace_table
        self.trace_table *= self.trace_factor * self.gamma
        
