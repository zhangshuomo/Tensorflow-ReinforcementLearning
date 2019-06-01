import time
import tensorflow as tf
import numpy as np
from DQN import DeepQnetwork
from Memory import Memory

class DQNExperienceReplay(DeepQnetwork):
	def __init__(self,n_features,n_actions,learning_rate=0.005,discount_factor=0.9,epsilon=0.9,
			memory_space=500,batch_size=32,steps_update=500,epsilon_increment=False,experience_replay=True):
		self.ex_r = experience_replay	# the reason is written down below
		super(DQNExperienceReplay,self).__init__(n_features,n_actions,learning_rate,discount_factor,epsilon,memory_space,batch_size,
						steps_update,epsilon_increment)
		if self.ex_r:
			self.memory = Memory(memory_space)

	def construct_networks(self):
		w_init = tf.random_normal_initializer(0.,0.3)
		b_init = tf.constant_initializer(0.1)

		c_name1 = ['evaluate',tf.GraphKeys.GLOBAL_VARIABLES]
		self.s_1 = tf.placeholder(tf.float32,[None,self.n_features])
		self.q_target = tf.placeholder(tf.float32,[None,self.n_actions])

		w1_1 = tf.get_variable('w1_1',shape=[self.n_features,10],initializer=w_init,collections=c_name1)
		b1_1 = tf.get_variable('b1_1',shape=[10],initializer=b_init,collections=c_name1)
		l1_1 = tf.nn.relu(tf.matmul(self.s_1,w1_1) + b1_1)
		w1_2 = tf.get_variable('w1_2',shape=[10,self.n_actions],initializer=w_init,collections=c_name1)
		b1_2 = tf.get_variable('b1_2',shape=[self.n_actions],initializer=b_init,collections=c_name1)
		self.l1_2 = tf.matmul(l1_1,w1_2) + b1_2
		if self.ex_r:	
		# when calling the construct function of base class, the function would be called first, so the attribute is missing.
			self.weights = tf.placeholder(tf.float32,[self.batch_size,1])
			self.abs_error = tf.reduce_sum(tf.abs(self.l1_2-self.q_target),axis=1)	# TD error
			self.loss = tf.reduce_mean(self.weights * tf.reduce_mean(tf.square(self.l1_2-self.q_target),axis=1)) 
		else:
			self.loss = tf.reduce_mean(tf.square(self.l1_2-self.q_target))
		self.train_op = tf.train.RMSPropOptimizer(self.alpha).minimize(self.loss)
		
		c_name2 = ['predictive',tf.GraphKeys.GLOBAL_VARIABLES]
		self.s_2 = tf.placeholder(tf.float32,[None,self.n_features])

		w2_1 = tf.get_variable('w2_1',shape=[self.n_features,10],initializer=w_init,collections=c_name2)
		b2_1 = tf.get_variable('b2_1',shape=[10],initializer=b_init,collections=c_name2)
		l2_1 = tf.nn.relu(tf.matmul(self.s_2,w2_1) + b2_1)
		w2_2 = tf.get_variable('w2_2',shape=[10,self.n_actions],initializer=w_init,collections=c_name2)
		b2_2 = tf.get_variable('b2_2',shape=[self.n_actions],initializer=b_init,collections=c_name2)
		self.l2_2 = tf.matmul(l2_1,w2_2) + b2_2

	def add_to_memory(self,state,action,reward,next_state):
		if self.ex_r:
			self.memory.add_to_memory(np.hstack([state,[action,reward],next_state]))
			# print(self.memory.data)	#
		else:
			super(DQNExperienceReplay,self).add_to_memory(state,action,reward,next_state)
	
	def learn(self):
		if 'begin' not in globals():
			globals()['begin']='begin to learn'
			print(globals()['begin'],time.ctime())
		if self.learning_times % self.steps_update == 0:
			c1 = tf.get_collection('evaluate')
			c2 = tf.get_collection('predictive')
			for item1, item2 in zip(c1, c2):
				self.sess.run(tf.assign(item2,item1))

		indices, weights = np.zeros(self.batch_size,dtype=np.int32), np.zeros((self.batch_size,),dtype=np.float32)
		samples = np.zeros((self.batch_size,2*self.n_features+2),dtype=np.float32)
		if self.ex_r:
			indices, weights, samples = self.memory.sample(self.batch_size)
		else:
			if self.memory_counter < self.memory.shape[0]:
				index = np.random.choice(self.memory_counter,size=self.batch_size)
			else:
				index = np.random.choice(self.memory.shape[0],size=self.batch_size)
			samples = self.memory[index,:]
		l1_output, l2_output = self.sess.run([self.l1_2,self.l2_2],feed_dict={self.s_1:samples[:,:self.n_features],
							self.s_2:samples[:,-self.n_features:]})
		q_target = l1_output
		actions = samples[:,self.n_features].astype(np.int32)
		rewards = samples[:,self.n_features+1]
		q_target[list(range(len(q_target))),actions] = rewards + self.gamma * np.max(l2_output,axis=1)
		if self.ex_r:
			abs_error = self.sess.run(self.abs_error,feed_dict={self.s_1:samples[:,:self.n_features],self.q_target:q_target})
			_,l = self.sess.run([self.train_op,self.loss],feed_dict={self.s_1:samples[:,:self.n_features],self.q_target:q_target,
										self.weights:weights[:,np.newaxis]})	###
			self.memory.update(indices,abs_error)
		else:
			_,l = self.sess.run([self.train_op,self.loss],feed_dict={self.s_1:samples[:,:self.n_features],self.q_target:q_target})
		if self.epsilon_increment and self.epsilon < self.epsilon_max:
			self.epsilon += self.epsilon_increment
		self.learning_times +=1


