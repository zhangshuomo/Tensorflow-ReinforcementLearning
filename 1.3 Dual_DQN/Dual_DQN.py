import tensorflow as tf
from DQN import DeepQnetwork

class Dual_DQN(DeepQnetwork):
	def __init__(self,n_features,n_actions,learning_rate=0.01,discount_factor=0.9,epsilon=0.9,
			memory_space=500,batch_size=32,steps_update=300,epsilon_increment=False,dual=True):
		self.dual = dual
		super(Dual_DQN,self).__init__(n_features,n_actions,learning_rate,discount_factor,epsilon,
				memory_space,batch_size,steps_update,epsilon_increment)

	def construct_networks(self):
		if self.dual == False:
			super().construct_networks()
		else:
			w_init = tf.random_normal_initializer(0.,0.3)
			b_init = tf.constant_initializer(0.1)

			c_name1 = ['evaluate',tf.GraphKeys.GLOBAL_VARIABLES]
			self.s_1 = tf.placeholder(tf.float32,[None,self.n_features])
			self.q_target = tf.placeholder(tf.float32,[None,self.n_actions])

			w1_1 = tf.get_variable('w1_1',shape=[self.n_features,10],initializer=w_init,collections=c_name1)
			b1_1 = tf.get_variable('b1_1',shape=[10],initializer=b_init,collections=c_name1)
			l1_1 = tf.nn.relu(tf.matmul(self.s_1,w1_1) + b1_1)
			w1_v = tf.get_variable('w1_v',shape=[10,1],initializer=w_init,collections=c_name1)
			b1_v = tf.get_variable('b1_v',shape=[1],initializer=b_init,collections=c_name1)
			w1_a = tf.get_variable('w1_a',shape=[10,self.n_actions],initializer=w_init,collections=c_name1)
			b1_a = tf.get_variable('b1_a',shape=[self.n_actions],initializer=b_init,collections=c_name1)
			self.l1_v = tf.matmul(l1_1,w1_v) + b1_v
			self.l1_a = tf.matmul(l1_1,w1_a) + b1_a
			self.l1_2 = self.l1_v + (self.l1_a - tf.reduce_mean(self.l1_a,axis=1,keep_dims=True))
		
			self.loss = tf.losses.mean_squared_error(self.l1_2,self.q_target)
			self.train_op = tf.train.RMSPropOptimizer(self.alpha).minimize(self.loss)
		
			c_name2 = ['predictive',tf.GraphKeys.GLOBAL_VARIABLES]
			self.s_2 = tf.placeholder(tf.float32,[None,self.n_features])

			w2_1 = tf.get_variable('w2_1',shape=[self.n_features,10],initializer=w_init,collections=c_name2)
			b2_1 = tf.get_variable('b2_1',shape=[10],initializer=b_init,collections=c_name2)
			l2_1 = tf.nn.relu(tf.matmul(self.s_2,w2_1) + b2_1)
			w2_v = tf.get_variable('w2_v',shape=[10,1],initializer=w_init,collections=c_name2)
			b2_v = tf.get_variable('b2_v',shape=[1],initializer=b_init,collections=c_name2)
			w2_a = tf.get_variable('w2_a',shape=[10,self.n_actions],initializer=w_init,collections=c_name2)
			b2_a = tf.get_variable('b2_a',shape=[self.n_actions],initializer=b_init,collections=c_name2)
			self.l2_v = tf.matmul(l2_1,w2_v) + b2_v
			self.l2_a = tf.matmul(l2_1,w2_a) + b2_a
			self.l2_2 = self.l2_v + (self.l2_a - tf.reduce_mean(self.l2_a,axis=1,keep_dims=True))

