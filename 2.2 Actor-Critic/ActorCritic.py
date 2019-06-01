import tensorflow as tf
import numpy as np

class ActorCritic(object):
	def __init__(self, n_features, n_actions=None, continuous=False, action_bounds=None,discount_factor=0.9, actor_learning_rate=0.001,  							critic_learning_rate=0.01):
		self.n_features = n_features
		self.n_actions = n_actions
		self.continuous = continuous
		self.gamma = discount_factor
		self.a_lr = actor_learning_rate
		self.c_lr = critic_learning_rate
		self.sess = tf.Session()
		
		self.s = tf.placeholder(tf.float32, [None,self.n_features])
		########################## actor ###############################
		self.td = tf.placeholder(tf.float32, None)

		if self.continuous:
			self.action_bounds = action_bounds; mini, maxi = self.action_bounds[0], self.action_bounds[1]
			self.action_chosen = tf.placeholder(tf.float32,None)
			a_l = tf.layers.dense(self.s,20,tf.nn.relu,kernel_initializer=tf.random_normal_initializer(0.,0.1),
					bias_initializer=tf.constant_initializer(0.1))
			mu = tf.layers.dense(a_l,1,tf.nn.sigmoid,kernel_initializer=tf.random_normal_initializer(0.,0.1),
					bias_initializer=tf.constant_initializer(0.1))
			sigma = tf.layers.dense(a_l,1,tf.nn.softplus,kernel_initializer=tf.random_normal_initializer(0.,0.1),
					bias_initializer=tf.constant_initializer(0.54))
			self.action_distribution = tf.distributions.Normal(0.5*(mini+maxi)*mu[0][0],sigma[0][0]*(maxi-mini)/6) 
										# 3 sigma falls into the feasible interval
			self.action = tf.clip_by_value(self.action_distribution.sample(1), mini, maxi)
			self.actor_loss = - self.td * self.action_distribution.log_prob(self.action_chosen) 
						
		else:	
			self.action_chosen = tf.placeholder(tf.int32, None)
			a_l = tf.layers.dense(self.s,20,tf.nn.relu,kernel_initializer=tf.random_normal_initializer(0.,0.1),
					bias_initializer=tf.constant_initializer(0.1))
			self.action_prob = tf.layers.dense(a_l,self.n_actions,tf.nn.softmax,
							kernel_initializer=tf.random_normal_initializer(0.,0.1),
							bias_initializer=tf.constant_initializer(0.1))

			action_chosen_prob = tf.reduce_sum(self.action_prob * tf.one_hot([self.action_chosen],self.n_actions))
			self.actor_loss = - self.td * tf.log(action_chosen_prob)

		self.actor_train_op = tf.train.AdamOptimizer(self.a_lr).minimize(self.actor_loss)
		########################## critic ###############################
		self.s_ = tf.placeholder(tf.float32,None)
		self.r = tf.placeholder(tf.float32,None)

		c_l = tf.layers.dense(self.s,20,tf.nn.relu,kernel_initializer=tf.random_normal_initializer(0.,0.1),
					bias_initializer=tf.constant_initializer(0.1))
		self.pred_v = tf.layers.dense(c_l,1,kernel_initializer=tf.random_normal_initializer(0.,0.1),
					bias_initializer=tf.constant_initializer(0.1))
		
		self.critic_loss = tf.square(self.r + self.gamma * self.s_ - self.pred_v)
		self.critic_train_op = tf.train.AdamOptimizer(self.c_lr).minimize(self.critic_loss)
		#################################################################
		self.sess.run(tf.global_variables_initializer())

	def choose_action(self, observation):
		observation = observation.reshape((1,self.n_features))
		if self.continuous:
			action = self.sess.run(self.action,{self.s:observation})
		else:
			action_prob = self.sess.run(self.action_prob,{self.s:observation})
			action_prob = action_prob.flatten()
			action = np.random.choice(self.n_actions,p=action_prob)
		return action

	def learn(self,observation,action,reward,observation_):
		observation = observation.reshape((1,self.n_features))
		observation_ = observation_.reshape((1,self.n_features))
		if not self.continuous:
			action = int(action)
		v_of_s = self.sess.run(self.pred_v,{self.s:observation})[0][0]
		v_of_s_ = self.sess.run(self.pred_v,{self.s:observation_})[0][0]
		td_error = reward + self.gamma * v_of_s_ - v_of_s
		_,a_l = self.sess.run([self.actor_train_op,self.actor_loss],{self.s:observation,self.td:td_error,self.action_chosen:action})
		_,c_l = self.sess.run([self.critic_train_op,self.critic_loss],{self.s:observation,self.r:reward,self.s_:v_of_s_})

