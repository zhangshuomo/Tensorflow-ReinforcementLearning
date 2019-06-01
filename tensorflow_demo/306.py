import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x=np.random.uniform(-1,1,1000)[:,None]
y=np.square(x)+np.random.normal(0,0.1,x.shape)
trainx,testx=np.split(x,[800])
trainy,testy=np.split(y,[800])

tf_x=tf.placeholder(tf.float32,trainx.shape)
tf_y=tf.placeholder(tf.float32,trainy.shape)

dataset=tf.data.Dataset.from_tensor_slices((tf_x,tf_y))
dataset=dataset.shuffle(1000)
dataset=dataset.repeat(3)
dataset=dataset.batch(32)
iterator=dataset.make_initializable_iterator()

bx,by=iterator.get_next()
l1=tf.layers.dense(bx,10,tf.nn.relu)
out=tf.layers.dense(l1,1)
loss=tf.losses.mean_squared_error(out,by)
train_op=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess=tf.Session()
sess.run([tf.global_variables_initializer(),iterator.initializer],feed_dict={tf_x:trainx,tf_y:trainy})
#fig,ax=plt.subplots(1,1)
#losses={'train':[],'test':[]}
for i in range(201):
	try:
		_,trainl = sess.run([train_op,loss])
		if i%5 == 0:
			testl=sess.run(loss,feed_dict={bx:testx,by:testy})
			#losses['train'].append(trainl)
			#losses['test'].append(testl)
			print('The train loss is %.4f, and the test loss is %.4f'%(trainl,testl))
	except tf.errors.OutOfRangeError:
		print('completed!')
		break
#ax.plot(losses['train'],label='train')
#ax.legend(loc='best')
#ax.plot(losses['test'],label='test')
#ax.legend(loc='best')
#plt.show()
