import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

ideas=5
LR=0.0001
samples=32

a=(np.random.rand(samples)+1)[:,np.newaxis]
x=np.linspace(-2,2,20)
a_=np.dot(a,np.ones(20).reshape(1,20))
y=a_*(x**2)+a_-1	# the dimension of y is (samples=32,20)

#fig,ax=plt.subplots(1,1)
#for i in range(3):
#	plt.plot(x,y[i],label='line %d'%i)
#	ax.legend(loc='best')
#plt.show()

with tf.variable_scope('Generator'):
	G_in=tf.placeholder(tf.float32,[None,ideas])
	G_layer=tf.layers.dense(G_in,40,tf.nn.relu)
	G_out=tf.layers.dense(G_layer,20)

with tf.variable_scope('Discriminator'):
	D_in=tf.placeholder(tf.float32,[None,20])
	D_layer=tf.layers.dense(D_in,60,tf.nn.relu,name='D_layer')
	D_out=tf.layers.dense(D_layer,1,tf.nn.sigmoid,name='D_out')

	D_layer_2=tf.layers.dense(G_out,60,tf.nn.relu,name='D_layer',reuse=True)
	D_out_2=tf.layers.dense(D_layer_2,1,tf.nn.sigmoid,name='D_out',reuse=True)

loss_1=-tf.reduce_sum(tf.log(D_out)+tf.log(1-D_out_2))/(2*samples)
loss_2=tf.reduce_mean(tf.log(1-D_out_2))
train_op_G=tf.train.AdamOptimizer(LR).minimize(loss_2,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Generator'))
train_op_D=tf.train.AdamOptimizer(LR).minimize(loss_1,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='Discriminator'))
sess=tf.Session()
sess.run(tf.global_variables_initializer())

fig=plt.figure()
plt.ion()
for step in range(1000):
	x_in=np.random.randn(samples,ideas)
	for step_D in range(5):
		D_l,_=sess.run([loss_1,train_op_D],feed_dict={D_in:y,G_in:x_in})
	for step_G in range(50):
		result,G_l,_=sess.run([G_out,loss_2,train_op_G],feed_dict={G_in:x_in})
	if step%20 == 0:
		plt.cla()
		plt.plot(x,y[0],'k-',lw=7,label='objective')
		plt.plot(x,result[0],'r-',lw=5,label='result')
		plt.pause(0.02)
		plt.legend(loc='best')
plt.ioff()
plt.show()

