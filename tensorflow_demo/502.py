# x is a 2-d array, notify x[:,0]!=x[:][0]
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


batch_size=64
epoch=12
LR=0.03
hidden_layers=8

tf.set_random_seed(1)
np.random.seed(1)

w_init=tf.random_normal_initializer(0.,.1)
b_init=tf.constant_initializer(-0.2)

x=np.linspace(-7,10,2000)[:,np.newaxis]
noise=np.random.normal(0,2,size=x.shape)
y=np.square(x)-5+noise 
train_data=np.hstack([x,y])

test_x=np.linspace(-7,10,200)[:,np.newaxis]
noise=np.random.normal(0,2,size=test_x.shape)
test_y=np.square(test_x)-5+noise    # forget to minus 5 !!
test_data=np.hstack([test_x,test_y])
np.random.shuffle(test_data)

tf_x=tf.placeholder(tf.float32,[None,1])
tf_y=tf.placeholder(tf.float32,[None,1])
tf_is_training=tf.placeholder(tf.bool,None)

class Net(object):
	def __init__(self,is_bn):
		self.is_bn=is_bn
		self.pre_activation=[tf_x]
		if self.is_bn:
			self.layer_input=[tf.layers.batch_normalization(tf_x,training=tf_is_training)]
		else:
			self.layer_input=[tf_x]
		for n in range(hidden_layers):
			self.layer_input.append(self.add_layer(self.layer_input[-1],10,tf.nn.tanh))
		out=tf.layers.dense(self.layer_input[-1],1,kernel_initializer=w_init,bias_initializer=b_init)
		self.loss=tf.losses.mean_squared_error(out,tf_y)
		update_op=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_op):
			self.train_op=tf.train.AdamOptimizer(LR).minimize(self.loss)

	def add_layer(self,in_put,output_size,activation=None):
		l=tf.layers.dense(in_put,output_size,kernel_initializer=w_init,bias_initializer=b_init)
		if self.is_bn:
			l=tf.layers.batch_normalization(l,momentum=0.4,training=tf_is_training) # forget to batch normalization !!
		self.pre_activation.append(l)
		if not activation is None:
			l=activation(l)
		return l

Nets=[Net(True),Net(False)]
sess=tf.Session()
sess.run(tf.global_variables_initializer())

fig,axes=plt.subplots(4,hidden_layers+1)
plt.ion()

def plot_histogram(*unkwargs): # input order is 'pre_ac,pre_ac_bn,layer_in,layer_in_bn'
	labels=['pre-ac','pre-ac with bn','layers input', 'layers input with bn']
	for row in range(len(unkwargs)):	
		for column,ax,variable in zip(range(hidden_layers+1),axes[row],unkwargs[row]):
			ax.clear()
			if column == 0:
				x_range=[-7,10]
				ax.set_ylabel(labels[row])
			else:
				if row<=1: x_range=[-4,4]
				else: x_range=[-1,1]
			ax.hist(variable.flatten(),range=x_range,bins=10)
			ax.set_xticks(())
			ax.set_yticks(())
	plt.pause(0.1)
	plt.show()

losses=[[],[]]
for e_num in range(epoch):
	step=0
	np.random.shuffle(train_data)
	in_epoch=True
	while in_epoch:
		bs,bf=step*batch_size%len(train_data),(step+1)*batch_size%len(train_data)
		if bf<bs:
			in_epoch=False
			bf=len(train_data)
		train_x,train_y=train_data[bs:bf,0:1],train_data[bs:bf,1:2]
		sess.run([Nets[0].train_op,Nets[1].train_op],feed_dict={tf_x:train_x,tf_y:train_y,tf_is_training:True})
		step+=1
		if step==1:
			l,l_bn,pre_ac,pre_ac_bn,layer_in,layer_in_bn=sess.run([Nets[1].loss,Nets[0].loss,Nets[1].pre_activation,
Nets[0].pre_activation,Nets[1].layer_input,Nets[0].layer_input],feed_dict={tf_x:test_data[:,0:1],tf_y:test_data[:,1:2],tf_is_training:False})
			plot_histogram(pre_ac,pre_ac_bn,layer_in,layer_in_bn)
			# print(test_data[:,0:1]==test_data[:][0][:,np.newaxis]) !!
			losses[0].append(l)
			losses[1].append(l_bn)
plt.ioff()
figure=plt.figure(2)
plt.plot(losses[0],label='without batch normalization')
plt.plot(losses[1],label='with batch normalizetion')
plt.ylim((0,2000))
plt.legend(loc='best')
plt.show()
