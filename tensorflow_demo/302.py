import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

b=np.ones((100,2))
x0=np.random.normal(2*b,1)
y0=np.zeros(100)
x1=np.random.normal(-2*b,1)
y1=np.ones(100)
x=np.vstack((x0,x1))
y=np.hstack((y0,y1))

fig=plt.figure()
ax=plt.subplot(111)
ax.scatter(x[:,0],x[:,1],c=y,cmap='RdYlGn')
plt.show()

tf_x=tf.placeholder(tf.float32,x.shape)
tf_y=tf.placeholder(tf.int64,y.shape)

l1=tf.layers.dense(tf_x,10,tf.nn.relu)
out=tf.layers.dense(l1,2)
result=tf.argmax(out,1)

loss=tf.losses.sparse_softmax_cross_entropy(logits=out,labels=tf_y)
accuracy=tf.reduce_mean(tf.cast(tf.equal(result,tf_y),tf.float32))
optimizer=tf.train.GradientDescentOptimizer(0.05)
train_op=optimizer.minimize(loss)

plt.ion()

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(100):
	_,l,r = sess.run([train_op,loss,result],feed_dict={tf_x:x,tf_y:y})
	if i % 2 == 0:
		plt.cla()
		plt.scatter(x[:,0],x[:,1],c=r,cmap='RdYlGn')
		plt.text(2,-4,'The loss is %.4f'%l,fontdict={'size':10,'color':'black'})
		plt.pause(0.1)
plt.ioff()
plt.show()
