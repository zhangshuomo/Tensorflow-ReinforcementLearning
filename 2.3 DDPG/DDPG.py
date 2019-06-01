import tensorflow as tf
import numpy as np

class DDPG(object):
	def __init__(self, feature_dim, action_dim, action_bound, actor_learning_rate=0.001, critic_learning_rate=0.002, 
			gamma=0.9, memory_space=10000, batch_size=32, soft=True, copy_rate=0.01, steps_update=None):
		self.feature_dim = feature_dim
		self.action_dim = action_dim
		self.action_bound = action_bound
		self.memory = np.zeros((memory_space,2*self.feature_dim+self.action_dim+1))
		self.batch_size = batch_size
		self.soft = soft
		if self.soft:
			self.copy_rate = copy_rate
		else:
			self.steps_update = steps_update
			self.counter = 0

		self.s = tf.placeholder(tf.float32,[None,self.feature_dim])
		self.a = tf.placeholder(tf.float32,[None,self.action_dim])
		self.r = tf.placeholder(tf.float32,[None,1])
		self.s_ = tf.placeholder(tf.float32,[None,self.feature_dim])

		with tf.variable_scope('action_target'):
			a_t_l1 = tf.layers.dense(self.s_,30,tf.nn.relu,trainable=False)
			a_t = tf.layers.dense(a_t_l1,self.action_dim, tf.nn.tanh, trainable=False) * self.action_bound
		s_a_t = tf.concat([self.s_, a_t], axis=1)
		with tf.variable_scope('q_target'):
			q_t_l1 = tf.layers.dense(s_a_t,30,tf.nn.relu,trainable=False)
			q_t = tf.layers.dense(q_t_l1, 1, trainable=False)
		with tf.variable_scope('action_eval'):	
			a_e_l1 = tf.layers.dense(self.s,30,tf.nn.relu)
			self.a_e = tf.layers.dense(a_e_l1, self.action_dim, tf.nn.tanh) * self.action_bound
		s_a = tf.concat([self.s,self.a],axis=1)		# for the training of the critic
		s_a_ = tf.concat([self.s,self.a_e],axis=1)	# for the training of the actor
		with tf.variable_scope('q_eval'):
			q_e_l1 = tf.layers.dense(s_a,30,tf.nn.relu, name='q_e_l1')
			q_e = tf.layers.dense(q_e_l1,1,name='q_e')
			q_e_l1_ = tf.layers.dense(s_a_,30,tf.nn.relu,name='q_e_l1',reuse=True)
			q_e_ = tf.layers.dense(q_e_l1_,1,name='q_e',reuse=True)
		critic_loss = tf.losses.mean_squared_error(self.r + gamma * q_t, q_e)	# changed, perhaps a bug
		self.critic_train_op = tf.train.AdamOptimizer(critic_learning_rate).minimize(critic_loss,var_list=
tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='q_eval'))
		actor_loss = -tf.reduce_mean(q_e_)	# It is 'reduce_mean' instead of 'reduce_sum' here
		self.actor_train_op = tf.train.AdamOptimizer(actor_learning_rate).minimize(actor_loss,
						var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='action_eval'))
		if self.soft:
			self.assign_op = [tf.assign(i,self.copy_rate*j+(1-self.copy_rate)*i) for i,j in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='action_target'),	 tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='action_eval'))]		# the troublesome bug is hidden here !!!
			assign_op_q = [tf.assign(i,self.copy_rate*j+(1-self.copy_rate)*i) for i,j in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='q_target'),tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='q_eval'))]
			self.assign_op.extend(assign_op_q)
		else:
			self.assign_op = [tf.assign(i,j) for i,j in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='action_target'),tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='action_eval'))]
			assign_op_q = [tf.assign(i,j) for i,j in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='q_target'),tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='q_eval'))]
			self.assign_op.extend(assign_op_q)
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

	def add_to_memory(self, observation, action, reward, observation_):
		if not hasattr(self,'memory_counter'):
			self.memory_counter = 0
		transition = np.hstack([observation,[action,reward],observation_])
		index = self.memory_counter % self.memory.shape[0]
		self.memory[index] = transition
		self.memory_counter += 1

	def choose_action(self, observation):
		if not hasattr(self,'std'):
			self.std = 3.
		observation = observation[np.newaxis,:]
		action = self.sess.run(self.a_e,{self.s:observation})[0]
		action = np.clip(np.random.normal(action, self.std),-self.action_bound,self.action_bound)
		return action	# this action is a vector

	def learn(self):
		assert self.memory_counter > self.memory.shape[0]
		indices = np.random.choice(self.memory.shape[0], size=self.batch_size)
		samples = self.memory[indices,:]
		self.sess.run(self.critic_train_op,{self.s:samples[:,:self.feature_dim],self.a:samples[:,self.feature_dim:
self.feature_dim+self.action_dim],self.r:samples[:,-self.feature_dim-1:-self.feature_dim],self.s_:samples[:,-self.feature_dim:]})
		self.sess.run(self.actor_train_op,{self.s:samples[:,:self.feature_dim]})
		if self.soft:
			self.sess.run(self.assign_op)
		else:
			if self.counter % self.steps_update == 0:
				self.sess.run(self.assign_op)
			self.counter += 1
		self.std *= 0.9995
		
		

		
