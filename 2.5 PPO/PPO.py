import tensorflow as tf
import numpy as np
import gym

class PPO(object):
	def __init__(self,n_features,n_actions,actor_learning_rate,critic_learning_rate,epsilon,sess):
		self.n_features=n_features
		self.n_actions=n_actions
		self.a_lr=actor_learning_rate
		self.c_lr=critic_learning_rate
		self.epsilon=epsilon
		self._construct_network()
		self.sess=sess
		self.sess.run(tf.global_variables_initializer())

	def _construct_network(self):
		self.tf_state=tf.placeholder(tf.float32,[None,self.n_features])
		with tf.variable_scope('actor'):
			self.advantages=tf.placeholder(tf.float32,[None,1])
			self.actions=tf.placeholder(tf.int32,[None,1])
			self.old_policy_param=tf.placeholder(tf.float32,[None,1])
			actor_l1=tf.layers.dense(self.tf_state,70,activation=tf.nn.relu,name='layer_1')
			self.actor_out=tf.layers.dense(actor_l1,self.n_actions,activation=tf.nn.softmax,name='action_prob')
		with tf.variable_scope('critic'):
			self.objective_values=tf.placeholder(tf.float32,[None,1])
			critic_l1=tf.layers.dense(self.tf_state,30,activation=tf.nn.relu,name='layer_1')
			self.critic_out=tf.layers.dense(critic_l1,1,name='state_value')
		self.ratio=tf.reduce_sum(self.actor_out * tf.one_hot(tf.squeeze(self.actions),depth=self.n_actions), keepdims=True, axis=1) / self.old_policy_param	# tf_state, actions, old_policy_param([1.0,...])
		entropy=tf.reduce_sum(self.actor_out * tf.log(self.actor_out),axis=1, keepdims=True)
		self.actor_loss=-tf.reduce_mean(tf.minimum(self.ratio*self.advantages,tf.clip_by_value(self.ratio,1-self.epsilon,1+self.epsilon)*self.advantages)+0.5*entropy)	# notify the hyperparameter here
		self.train_actor_op=tf.train.AdamOptimizer(self.a_lr).minimize(self.actor_loss)	# tf_state, actions, old_policy_param, advantages
		self.critic_loss=tf.losses.mean_squared_error(self.critic_out,self.objective_values)
		self.train_critic_op=tf.train.AdamOptimizer(self.c_lr).minimize(self.critic_loss)	# tf_state, objective_values

	def chooseAction(self,s):
		s=s[np.newaxis,:]
		prob=self.sess.run(self.actor_out,feed_dict={self.tf_state:s})
		action=np.random.choice(list(range(self.n_actions)),p=prob.flatten())
		return action

