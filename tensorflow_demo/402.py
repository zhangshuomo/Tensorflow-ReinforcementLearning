#notify the "batch_size" when testing the model
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

time_steps=28
input_size=28
batch_size=32
output_size=10
LR=0.01

mnist=input_data.read_data_sets('./mnist',one_hot=True)
test_x=mnist.test.images[:2000]
test_y=mnist.test.labels[:2000]

tf_x=tf.placeholder(tf.float32,[None,time_steps*input_size])
tf_y=tf.placeholder(tf.int32,[None,output_size])

image=tf.reshape(tf_x,[-1,time_steps,input_size])
rnn_cells=tf.contrib.rnn.BasicLSTMCell(64)
#init_state=rnn_cells.zero_state(batch_size,dtype=tf.float32)
outputs,final_state=tf.nn.dynamic_rnn(rnn_cells,image,initial_state=None,dtype=tf.float32,time_major=False)
#'None' here would not restrict the number of input samples
output=tf.layers.dense(outputs[:,-1,:],10)
loss=tf.losses.softmax_cross_entropy(logits=output,onehot_labels=tf_y)
accuracy=tf.metrics.accuracy(predictions=tf.argmax(output,1),labels=tf.argmax(tf_y,1))[1]
train_op=tf.train.AdamOptimizer(LR).minimize(loss)

sess=tf.Session()
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
for step in range(15000):
	xs,ys=mnist.train.next_batch(batch_size)
	_,l=sess.run([train_op,loss],feed_dict={tf_x:xs,tf_y:ys})
	if step%200==0:
		accuracy_=sess.run(accuracy,feed_dict={tf_x:test_x,tf_y:test_y})
		print('step%d:|The training loss is %.4f|The testing accuracy is %.4f'%(step,l,accuracy_))
test10=mnist.test.images[:10]
pred=sess.run(output,feed_dict={tf_x:test10})
prediction=np.argmax(pred,axis=1)
print('The prediction result is',prediction)
true_result=np.argmax(mnist.test.labels[:10],axis=1)
print('The true result is ',true_result)
