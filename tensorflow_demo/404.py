import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

LR=0.001
Num_examples=5
batch_size=64

mnist=input_data.read_data_sets('./mnist',one_hot=True)

tf_x=tf.placeholder(tf.float32,[None,28*28])

encode1=tf.layers.dense(tf_x,128,tf.nn.relu)
encode2=tf.layers.dense(encode1,64,tf.nn.relu)
encode3=tf.layers.dense(encode2,12,tf.nn.relu)
encoded=tf.layers.dense(encode3,3)

decode1=tf.layers.dense(encoded,12,tf.nn.relu)
decode2=tf.layers.dense(decode1,64,tf.nn.relu)
decode3=tf.layers.dense(decode2,128,tf.nn.relu)
decoded=tf.layers.dense(decode3,28*28,tf.nn.sigmoid)

loss=tf.losses.mean_squared_error(decoded,tf_x)
train_op=tf.train.AdamOptimizer(LR).minimize(loss)
fig,ax=plt.subplots(2,Num_examples,figsize=(Num_examples,2))
ts=mnist.test.images[:Num_examples]
for i in range(Num_examples):
	ax[0][i].imshow(ts[i].reshape(28,28),cmap='gray')
	ax[0][i].set_xticks(());ax[0][i].set_yticks(())
plt.ion()
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(8000):
	xs,_=mnist.train.next_batch(batch_size)
	_,l=sess.run([train_op,loss],feed_dict={tf_x:xs})
	if step%100 ==0:
		print('Step:%d, the loss is %.4f'%(step,l))
		result=sess.run(decoded,feed_dict={tf_x:ts})
		for i in range(Num_examples):
			ax[1][i].clear()
			ax[1][i].imshow(result[i].reshape(28,28),cmap='gray')
			ax[1][i].set_xticks(());ax[1][i].set_yticks(())
			plt.pause(0.1)
plt.ioff()
plt.show()
