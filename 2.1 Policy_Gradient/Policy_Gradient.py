import numpy as np
import tensorflow as tf

class PolicyGradient(object):
	def __init__(self,n_features,n_actions,learning_rate=0.01,discount_factor=0.9):
		self.n_features = n_features
		self.n_actions = n_actions
		self.alpha = learning_rate
		self.gamma = discount_factor
		self._construct_networks()
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.observations, self.actions, self.returns = [], [], []
	
	def _construct_networks(self):
		n_of_units = 10

		self.obs = tf.placeholder(tf.float32,[None,self.n_features])
		self.act = tf.placeholder(tf.int32,[None,])	# bug 4,  shape(1,n) is not the same as (n,) !!
		self.ret = tf.placeholder(tf.float32,[None,])

		layer_1 = tf.layers.dense(self.obs, n_of_units, tf.nn.tanh, kernel_initializer= \
				tf.random_normal_initializer(0,0.3),bias_initializer=tf.constant_initializer(0.1))
		self.layer_2 = tf.layers.dense(layer_1, self.n_actions, tf.nn.softmax, kernel_initializer= \
				tf.random_normal_initializer(0,0.3),bias_initializer=tf.constant_initializer(0.1)) 
		r = tf.reduce_sum(tf.log(self.layer_2) * tf.one_hot(self.act,self.n_actions), axis=1)	#
		self.loss = - tf.reduce_mean(r * self.ret)	# bug 4: this bug has been wrongly changed!
		self.train_op = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)
		
	def choose_action(self,observation):
		observation = observation[np.newaxis,:]
		prob_of_actions = self.sess.run(self.layer_2,feed_dict={self.obs:observation})
		prob_of_actions = prob_of_actions.flatten()
		action = np.random.choice(range(self.n_actions),p=prob_of_actions)
		return action

	def add_to_memory(self,observation,action,reward):
		self.observations.append(observation)
		self.actions.append(action)
		self.returns.append(reward)

	def learn(self):
		for i in np.arange(-2,-len(self.returns)-1,-1):
			self.returns[i] += self.returns[i+1] * self.gamma
		self.observations = np.array(self.observations)
		self.returns = self.returns - np.mean(self.returns)	# add a baseline here
		self.returns = self.returns / np.std(self.returns)	
		self.returns = np.array(self.returns)
		# for l in (self.actions,self.returns):	# bug 1!
		# 	l = np.array(l)[np.newaxis,:]
		self.actions = np.array(self.actions)	
		# print(self.returns.shape,self.actions.shape)
		self.sess.run(self.train_op,feed_dict={self.ret:self.returns,
						self.act:self.actions,
						self.obs:self.observations})
		self.observations, self.actions, self.returns = [], [], []	# bug 3!
	
