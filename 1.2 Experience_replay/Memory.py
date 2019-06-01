import numpy as np

class Memory(object):
	epsilon = 0.0001
	alpha = 1
	beta = 1
	abs_error_max = 1.0
	def __init__(self,capacity):
		self.data_pointer = 0
		self.capacity = capacity
		self.sumtree = np.zeros(2*self.capacity-1,dtype=np.float32)
		self.data= np.empty(shape=self.capacity,dtype=object)

	def add_to_memory(self,transition):
		self.data[self.data_pointer] = transition
		if (self.sumtree[-self.capacity:]==0).all():
			max_priority = self.abs_error_max
		else:
			max_priority = np.max(self.sumtree[-self.capacity:])
		tree_idx = self.data_pointer + self.capacity - 1
		diff = max_priority - self.sumtree[tree_idx]
		self.sumtree[tree_idx] = max_priority
		while tree_idx != 0:
			tree_idx = (tree_idx - 1) // 2
			self.sumtree[tree_idx] += diff
		self.data_pointer += 1
		if self.data_pointer == self.capacity:
			self.data_pointer = 0
	
	def _obtain_data(self,v):
		tree_idx = 0
		while True:
			if 2*tree_idx+1 >= len(self.sumtree):
				return tree_idx, self.sumtree[tree_idx], self.data[tree_idx-self.capacity+1] 
				# the index, the priority and the corresponding transition 
			else:
				if self.sumtree[2*tree_idx+1] > v:	# bug 1
					tree_idx = 2 * tree_idx + 1
				else:
					v -= self.sumtree[2*tree_idx+1]	# bug 2
					tree_idx = 2 * tree_idx + 2
			
	def sample(self,n):
		sum_of_p = self.sumtree[0]
		interval = sum_of_p / n
		indices = np.zeros((n,),dtype=np.int32)
		transitions = []
		priorities = np.zeros((n,),dtype=np.float32)
		for i in range(n):
			v = np.random.uniform(i*interval,(i+1)*interval)
			index, priority, transition = self._obtain_data(v)
			indices[i] = index
			transitions.append(transition)
			priorities[i] = priority
		# print(transitions)
		transitions = np.array(transitions)
		# print(transitions)
		probabilities = priorities / sum_of_p
		weights = np.power(probabilities / np.min(probabilities), -self.beta)
		return indices, weights, transitions
		# return the positions of the transitions in the 'sumtree' ,the weights and the transitions themselves.
	
	def update(self, indices, new_priorities):
		new_priorities += self.epsilon
		new_priorities = np.power(new_priorities, self.alpha)
		new_priorities = np.minimum(new_priorities, np.array([self.abs_error_max]*len(new_priorities)))
		for index, new_priority in zip(indices, new_priorities):
			diff = new_priority - self.sumtree[index]
			self.sumtree[index] = new_priority
			while index != 0:
				index = (index - 1) // 2
				self.sumtree[index] += diff
		
if __name__ == '__main__':
	m = Memory(500)
	for i in range(600):
		m.add_to_memory(np.random.randint(25,size=(2,)))
	for i in range(m.capacity-1):
		print(i,m.sumtree[i])
	for i,j in enumerate(m.data):
		print(i+m.capacity-1,m.sumtree[i+m.capacity-1],j)
	i,w,t = m.sample(50)
	print('-'*50)
	print(i)
	print(w)
	print(t)
	l = []
	for s in range(len(i)):
		l.append(np.random.uniform())
	print('-'*50)
	print(l)
	m.update(i,np.array(l))
	print('-'*50)
	for i in range(m.capacity-1):
		print(i,m.sumtree[i])
	for i,j in enumerate(m.data):
		print(i+m.capacity-1,m.sumtree[i+m.capacity-1],j)
	i,w,t = m.sample(50)
	print('-'*50)
	print(i)
	print(w)
	print(t)

