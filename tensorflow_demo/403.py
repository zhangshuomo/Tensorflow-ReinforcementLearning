import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

time_step=10
LR=0.02
input_size=1
output_size=1
cell_size=16

tf_x=tf.placeholder(tf.float32,[None,time_step,input_size])
tf_y=tf.placeholder(tf.float32,[None,time_step,output_size])
cells=tf.contrib.rnn.BasicLSTMCell(cell_size)
init_s=cells.zero_state(batch_size=1,dtype=tf.float32)
output,final_s=tf.nn.dynamic_rnn(cells,tf_x,initial_state=init_s,time_major=False)#initial_state parameter !!
outputs=tf.layers.dense(tf.reshape(output,[-1,cell_size]),output_size)
out=tf.reshape(outputs,[-1,time_step,output_size])
loss=tf.losses.mean_squared_error(out,tf_y)
train_op=tf.train.AdamOptimizer(LR).minimize(loss)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
fig,ax=plt.subplots(1,1,figsize=(12,5))
plt.ion()
for step in range(60):
	start,end = step*np.pi,(step+1)*np.pi
	x = np.linspace(start,end,time_step)[np.newaxis,:,np.newaxis]
	np_x = np.sin(x)
	np_y = np.cos(x)
	if 'final_s_' in globals():
		feed={tf_x:np_x,tf_y:np_y,init_s:final_s_}
	else:
		feed={tf_x:np_x,tf_y:np_y}
	_,loss_,out_,final_s_= sess.run([train_op,loss,out,final_s],feed_dict=feed)
	ax.plot(x.squeeze(),np_y.squeeze(),'k-',label='target')
	ax.plot(x.squeeze(),out_.squeeze(),'r--',label='output')
	plt.pause(0.05)
plt.ioff()
plt.show()
	


