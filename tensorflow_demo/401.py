import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE=32
LEARNING_RATE=0.01


mnist=input_data.read_data_sets('./mnist',one_hot=True)
test_x=mnist.test.images[:2000]
test_y=mnist.test.labels[:2000]

tf_x=tf.placeholder(tf.float32,[None,28*28])
tf_y=tf.placeholder(tf.int32,[None,10])

image=tf.reshape(tf_x,[-1,28,28,1])
conv1=tf.layers.conv2d(image,filters=16,kernel_size=5,strides=1,padding='SAME',activation=tf.nn.relu)
max_pool1=tf.layers.max_pooling2d(conv1,pool_size=2,strides=2)
conv2=tf.layers.conv2d(max_pool1,filters=32,kernel_size=5,strides=1,padding='SAME',activation=tf.nn.relu)
max_pool2=tf.layers.max_pooling2d(conv2,pool_size=2,strides=2)
flat=tf.reshape(max_pool2,[-1,7*7*32])
output=tf.layers.dense(flat,10)

loss=tf.losses.softmax_cross_entropy(logits=output,onehot_labels=tf_y)
accuracy=tf.metrics.accuracy(labels=tf.argmax(tf_y,axis=1),predictions=tf.argmax(output,axis=1))[1]
train_op=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

sess=tf.Session()
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
for i in range(600):
	xs,ys=mnist.train.next_batch(BATCH_SIZE)
	_,l=sess.run([train_op,loss],feed_dict={tf_x:xs,tf_y:ys})
	if i%50==0:
		accuracy_=sess.run(accuracy,feed_dict={tf_x:test_x,tf_y:test_y})
		print('step%d|the training loss is %.4f|the testing accuracy is %.4f'%(i,l,accuracy_))

test10=mnist.test.images[:10]
output_=sess.run(output,feed_dict={tf_x:test10})
pred_result=np.argmax(output_,1)
true_result=np.argmax(mnist.test.labels[:10],1)
print('The first 10 samples in testing set:')
print('prediction result is',pred_result,)
print('true result is', true_result)

