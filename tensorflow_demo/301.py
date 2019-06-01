import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fig,ax=plt.subplots(1,1)
x=np.linspace(-1,1,100)[:,None]
noise=np.random.normal(0,0.1,size=x.shape)
y=np.square(x)+noise
ax.scatter(x,y)
plt.ion()

tf_x=tf.placeholder(tf.float32,x.shape)
tf_y=tf.placeholder(tf.float32,y.shape)

l1=tf.layers.dense(tf_x,10,tf.nn.relu)
pred=tf.layers.dense(l1,1)

loss=tf.losses.mean_squared_error(pred,tf_y)
optimizer=tf.train.GradientDescentOptimizer(0.5)
train_op=optimizer.minimize(loss)
sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
	_,l,prediction=sess.run([train_op,loss,pred],feed_dict={tf_x:x,tf_y:y})
	if i % 5 == 0:
		print(l)
		plt.cla()
		plt.scatter(x,y)
		plt.plot(x,prediction,'r-',lw=5)
		plt.text(0.5,0,'The loss is %.4f'%l,fontdict={'size':10,'color':'blue'})
		plt.pause(0.1)
plt.ioff()
plt.show()
		

