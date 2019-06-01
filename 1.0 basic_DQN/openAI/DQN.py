import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)

class DeepQnetwork(object):
	def __init__(self,n_features,n_actions,learning_rate=0.01,discount_factor=0.9,epsilon=0.9,
			memory_space=500,batch_size=32,steps_update=300,epsilon_increment=False):
		self.n_features = n_features
		self.n_actions = n_actions
		self.alpha = learning_rate
		self.gamma = discount_factor
		# self.epsilon = epsilon
		self.memory = np.zeros((memory_space,2*n_features+2))
		self.batch_size = batch_size
		self.steps_update = steps_update
		self._construct_networks()
		self.sess=tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.learning_times = 0
		self.epsilon_increment = epsilon_increment
		if self.epsilon_increment:
			self.epsilon = 0
			self.epsilon_max = epsilon
		else:
			self.epsilon = epsilon

	def _construct_networks(self):
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
		
		self.loss = tf.losses.mean_squared_error(self.l1_2,self.q_target)
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
		state = state.squeeze()
		next_state = next_state.squeeze()
		if not hasattr(self,'memory_counter'): #
			self.memory_counter = 0
		transition = np.hstack([state,[action,reward],next_state]) #
		index = self.memory_counter % self.memory.shape[0]
		self.memory[index,:] = transition
		self.memory_counter += 1

	def choose_action(self,state):
		if np.random.uniform() > self.epsilon:
			action = np.random.choice(self.n_actions)
		else:
			l1_output = self.sess.run(self.l1_2,feed_dict={self.s_1:state[np.newaxis,:]})
			action = np.argmax(l1_output)
			# print(action)
		return action
		
	def learn(self):
		if self.learning_times % self.steps_update == 0:
			c1 = tf.get_collection('evaluate')
			c2 = tf.get_collection('predictive')
			for item1, item2 in zip(c1, c2):
				self.sess.run(tf.assign(item2,item1))
		if self.memory_counter < self.memory.shape[0]:
			index = np.random.choice(self.memory_counter,size=self.batch_size)
		else:
			index = np.random.choice(self.memory.shape[0],size=self.batch_size)
		samples = self.memory[index,:]
		# print(samples)
		l1_output, l2_output = self.sess.run([self.l1_2,self.l2_2],feed_dict={self.s_1:samples[:,:self.n_features],
							self.s_2:samples[:,-self.n_features:]})
		q_target = l1_output
		# print(q_target)
		actions = samples[:,self.n_features].astype(np.int32)
		# print(actions)
		rewards = samples[:,self.n_features+1]
		# print(rewards)
		q_target[list(range(len(q_target))),actions] = rewards + self.gamma * np.max(l2_output,axis=1)
		# print(q_target)
		_,l = self.sess.run([self.train_op,self.loss],feed_dict={self.s_1:samples[:,:self.n_features],self.q_target:q_target})
		if self.epsilon_increment and self.epsilon < self.epsilon_max:
			self.epsilon += self.epsilon_increment
		self.learning_times +=1

