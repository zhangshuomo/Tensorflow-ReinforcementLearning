import tensorflow as tf
import numpy as np

x=np.linspace(-1,1,100)[:,np.newaxis]
noise=np.random.normal(np.zeros(x.shape),.1)
y=np.square(x)+noise

with tf.variable_scope('inputs'):
	tf_x=tf.placeholder(tf.float32,x.shape,name='x')
	tf_y=tf.placeholder(tf.float32,y.shape,name='y')

with tf.variable_scope('layer'):
	l1=tf.layers.dense(tf_x,10,tf.nn.relu)
	output=tf.layers.dense(l1,1)

tf.summary.histogram('hidden_layer',l1)
tf.summary.histogram('output',output)

loss=tf.losses.mean_squared_error(output,tf_y)

tf.summary.scalar('losses:',loss)

optimizer=tf.train.GradientDescentOptimizer(0.5)
train_op=optimizer.minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
writer=tf.summary.FileWriter('logs/',sess.graph)
merge_op=tf.summary.merge_all()
for i in range(100):
	_,result = sess.run([train_op,merge_op],feed_dict={tf_x:x,tf_y:y})
	writer.add_summary(result,i)
