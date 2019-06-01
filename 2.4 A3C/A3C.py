import tensorflow as tf
import gym
import numpy as np

class ACNet(object):
	def __init__(self,name, n_feature,n_action,action_bound, sess, actor_learning_rate=None,critic_learning_rate=None,
beta=None,memory_space=None, gamma=None,center=None):
		self.name=name
		self.n_feature=n_feature
		self.n_action=n_action
		self.action_bound=action_bound
		self.center=center
		if self.center:
			self.gamma=gamma
			self.beta=beta
			self.AOptimizer=tf.train.RMSPropOptimizer(actor_learning_rate)
			self.COptimizer=tf.train.RMSPropOptimizer(critic_learning_rate)
			self.memory=np.zeros((memory_space,n_feature*2+n_action+1))
			self.memory_pointer=0
		self.sess=sess
		self._construct_network()
		self.sess.run(tf.global_variables_initializer())

	def _construct_network(self):
		initializer=tf.random_normal_initializer(0.,.1)
		self.state=tf.placeholder(tf.float32,[None,self.n_feature])
		self.action=tf.placeholder(tf.float32,[None,self.n_action])
		self.q_value=tf.placeholder(tf.float32,[None,1])
		with tf.variable_scope(self.name+'/Actor'):
			a_l1=tf.layers.dense(self.state,200,tf.nn.relu6,kernel_initializer=initializer,name='layer1')
			mu=tf.layers.dense(a_l1,self.n_action,tf.nn.tanh,kernel_initializer=initializer,name='action_mean')
			sigma=tf.layers.dense(a_l1,self.n_action,tf.nn.softplus,kernel_initializer=initializer,name='action_std')
		with tf.variable_scope(self.name+'/Critic'):
			c_l1=tf.layers.dense(self.state,100,tf.nn.relu6,kernel_initializer=initializer,name='layer1')
			self.value=tf.layers.dense(c_l1,1,name='value')
		mu*=self.action_bound
		sigma+=10**(-4)
		distribution=tf.distributions.Normal(mu,sigma)
		# parameters:
		self.actor_params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/Actor')
		self.critic_params=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name+'/Critic')
		# choose action op:
		self.a=tf.clip_by_value(distribution.sample(1),-self.action_bound,self.action_bound)	# state
		if self.center:
			# actor loss:
			actor_loss=-tf.reduce_mean(tf.stop_gradient(self.q_value-self.value)*distribution.log_prob(self.action)+self.beta * distribution.entropy())# q_value, state, action			
			self.actor_gradients=tf.gradients(actor_loss,self.actor_params)
			# critic loss:
			critic_loss=tf.reduce_mean(tf.square(self.q_value-self.value))	# q_value, state
			self.critic_gradients=tf.gradients(critic_loss,self.critic_params,name='critic_gradients')
			# train and pull operations
			self.train_op=[self.AOptimizer.apply_gradients(zip(self.actor_gradients,self.center.actor_params)), self.COptimizer.apply_gradients(zip(self.critic_gradients, self.center.critic_params))]
			self.pull_op=[tf.assign(a,c) for a,c in zip(self.actor_params,self.center.actor_params)]
			self.pull_op.extend([tf.assign(a,c) for a,c in zip(self.critic_params,self.center.critic_params)])

	def choose_action(self,s):
		action=self.sess.run(self.a, {self.state: s[np.newaxis,:]})[0][0]
		return action	# the shape is [1,]

	def add_to_memory(self, s, a, r, s_):
		transition=np.hstack([s,a,[r],s_])
		memory_space=self.memory.shape[0]
		assert self.memory_pointer<memory_space
		self.memory[self.memory_pointer,:]=transition
		self.memory_pointer+=1

	def learn(self):
		v_of_s_=self.sess.run(self.value, {self.state:self.memory[self.memory_pointer-1:self.memory_pointer,-self.n_feature:]})[0][0]	# bug 1: self.memory[-1:,-self.n_feature:]
		# print('v_s_:',v_of_s_)
		# print('memory:',self.memory)
		q_value=[[0]]*self.memory_pointer
		for i in range(self.memory_pointer):
			q_value[i]=[self.memory[self.memory_pointer-1-i,-self.n_feature-1]+self.gamma*v_of_s_]
			v_of_s_=q_value[i][0]	# bug 2	v_s_ spell error
		q_value.reverse()
		q_value=np.array(q_value)
		# print('q_value:',q_value)
		state=self.memory[:self.memory_pointer,:self.n_feature]
		action=self.memory[:self.memory_pointer,self.n_feature:self.n_feature+self.n_action]
		# print('state:',state)
		# print('action:',action)
		self.sess.run(self.train_op,{self.state:state,self.action:action,self.q_value:q_value})
		self.sess.run(self.pull_op)
		self.memory_pointer=0
		# print('----------------------------------------------------')

class A3C_Worker(object):
	total_episodes=2000
	total_steps=200
	epi=0
	rewards=[]
	sess=tf.Session()

	def __init__(self, name, game, a_lr=0.0001, c_lr=0.001, beta=0.01, iter_steps=10, discount_facor=0.9):
		self.name=name
		self.env=gym.make(game).unwrapped
		n_feature=self.env.observation_space.shape[0]
		n_action=self.env.action_space.shape[0]
		action_bound=self.env.action_space.high[0]
		if 'center' not in self.__class__.__dict__:
			self.__class__.center=ACNet('CenterUnit', n_feature, n_action, action_bound, self.__class__.sess)
		self.acnet=ACNet(self.name, n_feature,n_action, action_bound, self.__class__.sess, a_lr, c_lr, beta, iter_steps, discount_facor, self.__class__.center)
		self.iter_steps=iter_steps

	def work(self):
		while self.__class__.epi < self.__class__.total_episodes:
			state=self.env.reset()
			total_reward=0
			for step in range(self.__class__.total_steps):
				if self.name=='w1' and self.__class__.epi>1:	
					self.env.render()
				action=self.acnet.choose_action(state)
				state_, reward, done, info=self.env.step(action)
				total_reward+=reward
				self.acnet.add_to_memory(state, action, (reward+8)/8, state_)
				if step==self.__class__.total_steps-1: done=True
				if step%self.iter_steps==0 or done:
					self.acnet.learn()
				if done:
					if len(self.__class__.rewards)>0:
						total_reward=self.__class__.rewards[-1]*0.9+total_reward*0.1
					self.__class__.rewards.append(total_reward)
					print('{0:4} |epi:{1:4}, | total rewards:{2:7}'.format(self.name, self.__class__.epi+1,int(total_reward)))
				state=state_
			self.__class__.epi+=1

	@classmethod
	def moving_rewards(cls):
		return cls.rewards
