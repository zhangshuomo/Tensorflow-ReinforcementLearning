from PPO import PPO
import threading as td
import gym
import tensorflow as tf
import numpy as np

class PPO_Worker(object):
	update_material=[]
	moving_rewards=[]
	pointer=0
	n_of_PPO_Workers=0
	total_epi=1000
	epi=0
	appendEvent=td.Event()
	updateEvent=td.Event()
	update_thread_ended=False
	def __init__(self, game, name, actor_learning_rate=0.0002, critic_learning_rate=0.0003, memorysize=64, gamma=0.9, epsilon=0.2):
		self.env=gym.make(game).unwrapped
		self.name=name
		self.n_features=self.env.observation_space.shape[0]
		self.n_actions=self.env.action_space.n
		if 'PPO_unit' not in self.__class__.__dict__:
			self.__class__.sess=tf.Session()
			self.__class__.PPO_unit=PPO(self.n_features, self.n_actions, actor_learning_rate, critic_learning_rate, epsilon, self.__class__.sess)
			self.__class__.gamma=gamma
			self.__class__.appendEvent.set()
			self.__class__.updateEvent.clear()
		self.__class__.n_of_PPO_Workers+=1
		self.memory=np.zeros((memorysize, 2*self.n_features + 2))
		self.memory_pointer=0

	def work(self):	# the PPO_unit is read-only in this method
		while self.__class__.epi<self.__class__.total_epi:
			state=self.env.reset()
			total_reward=0
			step=0
			while True:
				self.__class__.appendEvent.wait()
				action=self.__class__.PPO_unit.chooseAction(state)
				state_, reward, done, info=self.env.step(action)
				if done:	reward=-20
				total_reward+=reward
				if step>=500: done=True
				self._add_to_memory(state, action, reward, state_)
				if step%self.memory.shape[0]==0 or done:
					self._add_to_material()
				if done:
					print('{0:3}|epi:{1:5}|total_reward:{2:5}'.format(self.name,self.__class__.epi+1, total_reward))
					if len(self.__class__.moving_rewards)!=0:
						total_reward=self.__class__.moving_rewards[-1]*0.9+total_reward*0.1
					self.__class__.moving_rewards.append(total_reward)
					break
				state=state_
				step+=1
			self.__class__.epi+=1
			
	def _add_to_memory(self, s, a, r, s_):
		assert self.memory_pointer<self.memory.shape[0]
		transition=np.concatenate((s,[a,r],s_))
		self.memory[self.memory_pointer]=transition
		self.memory_pointer+=1

	def _add_to_material(self):
		states=self.memory[:self.memory_pointer,:self.n_features]
		actions=self.memory[:self.memory_pointer,self.n_features:self.n_features+1].astype(np.int32)
		returns=[[0.]]*self.memory_pointer
		returns[0]=[self.__class__.gamma * self.__class__.sess.run(self.__class__.PPO_unit.critic_out,feed_dict={self.__class__.PPO_unit.tf_state:states[-1:]})[0][0]+self.memory[self.memory_pointer-1,-self.n_features-1]]
		for i in range(1,len(returns)):
			returns[i]=[self.memory[self.memory_pointer-1-i,-self.n_features-1]+self.__class__.gamma*returns[i-1][0]]
		returns.reverse()
		returns=np.array(returns)
		advantages=returns-self.__class__.sess.run(self.__class__.PPO_unit.critic_out,{self.__class__.PPO_unit.tf_state:states})
		old_policy_param=self.__class__.sess.run(self.__class__.PPO_unit.ratio,{self.__class__.PPO_unit.tf_state:states,self.__class__.PPO_unit.actions:actions,self.__class__.PPO_unit.old_policy_param:np.array([[1.]]*self.memory_pointer)})
		self.__class__.update_material.append(np.hstack([states, actions, old_policy_param, advantages, returns]))
		self.memory_pointer=0
		if len(self.__class__.update_material) >= self.__class__.n_of_PPO_Workers and not self.__class__.update_thread_ended:# modified	
			self.__class__.appendEvent.clear()	 
			self.__class__.updateEvent.set()
			# Deadlock: the updateParameters thread ends up earlier, resulting no effect of the set of updateEvent. Also, the appendEvent is cleared, making the other working threads still waiting.

	@classmethod
	def updateParameters(cls):	
		while cls.epi<cls.total_epi:
			cls.updateEvent.wait()
			material=np.vstack(cls.update_material)
			returns=material[:,-1:]
			advantages=material[:,-2:-1]
			old_policy_param=material[:,-3:-2]
			actions=material[:,-4:-3].astype(np.int32)
			states=material[:,:-4]
			for i in range(10): # notify the hyperparameter here
				cls.sess.run(cls.PPO_unit.train_actor_op,{cls.PPO_unit.tf_state:states,cls.PPO_unit.actions:actions,cls.PPO_unit.old_policy_param:old_policy_param,cls.PPO_unit.advantages:advantages})
			for j in range(20):# notify the hyperparameter here
				cls.sess.run(cls.PPO_unit.train_critic_op,{cls.PPO_unit.tf_state:states,cls.PPO_unit.objective_values:returns})
			cls.update_material.clear()
			cls.updateEvent.clear()
			cls.appendEvent.set()
		cls.update_thread_ended=True
	@classmethod
	def get_moving_rewards(cls):
		return cls.moving_rewards
