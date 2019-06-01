import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)
#the two codes above make some sense

LR=0.01

x=np.linspace(-1,1,20)[:,np.newaxis]
y=x+0.3*np.random.randn(*(x.shape))
test_x=x.copy()
test_y=test_x+0.3*np.random.randn(*(test_x.shape))

tf_x=tf.placeholder(tf.float32,[None,1])
tf_y=tf.placeholder(tf.float32,[None,1])
tf_rate=tf.placeholder(tf.float32)
is_training=tf.placeholder(tf.bool,None)

o1=tf.layers.dense(tf_x,300,tf.nn.relu)
o2=tf.layers.dense(o1,300,tf.nn.relu)
o_output=tf.layers.dense(o2,1)
o_loss=tf.losses.mean_squared_error(tf_y,o_output)
o_train_op=tf.train.AdamOptimizer(LR).minimize(o_loss)

d1=tf.layers.dense(tf_x,300,tf.nn.relu)
d1_=tf.layers.dropout(d1,rate=tf_rate,training=is_training)
d2=tf.layers.dense(d1_,300,tf.nn.relu)
d2_=tf.layers.dropout(d2,rate=tf_rate,training=is_training)
d_output=tf.layers.dense(d2,1)
d_loss=tf.losses.mean_squared_error(tf_y,d_output)
d_train_op=tf.train.AdamOptimizer(LR).minimize(d_loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
fig,ax=plt.subplots()
plt.ion()
for step in range(500):
	sess.run([o_train_op,d_train_op],feed_dict={tf_x:x,tf_y:y,tf_rate:0.5,is_training:True})
	if step%10==0:
		ax.clear()
		o_result,d_result,ol,dl=sess.run([o_output,d_output,o_loss,d_loss],feed_dict={tf_x:test_x,tf_y:test_y,tf_rate:0.5,is_training:False})
		ax.scatter(x,y,c='red',s=50,alpha=0.5,label='training data')
		ax.scatter(test_x,test_y,c='green',s=50,alpha=0.5,label='testing data')
		ax.plot(test_x,o_result,'r-',lw=3,label='overfitting')
		ax.plot(test_x,d_result,'b--',lw=3,label='drop out')
		ax.legend(loc='best')
		ax.text(0,-1.5,'The overfitting loss is %.4f'%ol,fontdict={'size':10,'color':'red'})
		ax.text(0,-1.3,'The dropout loss is %.4f'%dl,fontdict={'size':10,'color':'green'})
		ax.set_ylim((-2.5,2.5))
		plt.pause(0.1)
plt.ioff()
plt.show()


