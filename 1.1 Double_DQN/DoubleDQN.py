import DQN
import tensorflow as tf
import numpy as np

class DoubleDQN(DQN.DeepQnetwork):
	def __init__(self,n_features,n_actions,learning_rate=0.01,discount_factor=0.9,epsilon=0.9,
			memory_space=500,batch_size=32,steps_update=300,epsilon_increment=False,double=True):
		super(DoubleDQN,self).__init__(n_features,n_actions,learning_rate,discount_factor,epsilon,memory_space,batch_size,
						steps_update,epsilon_increment)
		self.double = double

	def learn(self):
		if not self.double:
			super(DoubleDQN,self).learn()
		else:
			if self.learning_times % self.steps_update == 0:
				params1 = tf.get_collection('evaluate')
				params2 = tf.get_collection('predictive')
				for i,j in zip(params1,params2):
					self.sess.run(tf.assign(j,i))
			if self.memory_counter < self.memory.shape[0]:
				batch_index = np.random.choice(self.memory_counter,size=self.batch_size)
			else:
				batch_index = np.random.choice(self.memory.shape[0],size=self.batch_size)
			samples = self.memory[batch_index,:]
			l1_2, l2_2 = self.sess.run([self.l1_2, self.l2_2],{self.s_1:samples[:,-self.n_features:],self.s_2:samples[:,-self.n_features:]})	# l1_2 and l2_2 are shaped as samples * q_values_of_actions
			action_chosen = np.argmax(l2_2,axis=1)
			q_value = l1_2[list(range(samples.shape[0])),action_chosen]
			state_q = self.sess.run(self.l1_2,{self.s_1:samples[:,:self.n_features]})
			actions = samples[:,self.n_features].astype(np.int32)
			rewards = samples[:,self.n_features+1]
			# print(type(state_q))
			state_q[list(range(samples.shape[0])),actions] = rewards + self.gamma * q_value
			self.learning_times += 1
			_,loss = self.sess.run([self.train_op,self.loss],{self.s_1:samples[:,:self.n_features],self.q_target:state_q})
			if self.epsilon_increment and self.epsilon < self.epsilon_max:
				self.epsilon += self.epsilon_increment
