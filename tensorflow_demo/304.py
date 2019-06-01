import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

LR=0.01
batch_size=32

x=np.linspace(-1,1,100)[:,np.newaxis]
noise=np.random.normal(np.zeros(x.shape),0.1)
y=np.power(x,2)+noise
fig=plt.figure(figsize=(10,5))
ax=plt.subplot(1,2,1)
ax.scatter(x,y)

class Net(object):
	def __init__(self,opt):
		self.x=tf.placeholder(tf.float32,[None,x.shape[1]])
		self.y=tf.placeholder(tf.float32,[None,y.shape[1]])
		l1=tf.layers.dense(self.x,10,tf.nn.relu)
		self.output=tf.layers.dense(l1,1)
		self.loss=tf.losses.mean_squared_error(self.output,self.y)
		self.train_step=opt.minimize(self.loss)

net_SGD=Net(tf.train.GradientDescentOptimizer(LR))
net_Momentum=Net(tf.train.MomentumOptimizer(LR,momentum=.9))
net_Adam=Net(tf.train.AdamOptimizer(LR))
net_RMSprop=Net(tf.train.RMSPropOptimizer(LR))

sess=tf.Session()
sess.run(tf.global_variables_initializer())
loss_dict={net_SGD:[],net_Momentum:[],net_Adam:[],net_RMSprop:[]}

for i in range(100):
	index=np.random.randint(0,x.shape[0],batch_size)
	xs=x[index]
	ys=y[index]
	for net in loss_dict:
		_,l = sess.run([net.train_step,net.loss],feed_dict={net.x:xs,net.y:ys})
		if i % 10 == 0:
			loss_dict[net].append(l)
names=['SGD','Momentum','Adam','RMSprop']
i=0
for net in loss_dict:
	ax=plt.subplot(122)
	ax.plot(loss_dict[net],label=names[i])
	i+=1
ax.legend(loc='best')
plt.show()

			
