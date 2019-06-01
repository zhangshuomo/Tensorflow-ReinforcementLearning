import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

true_value=[1.2,2.5]
initial_value=[2,4.5]
LR=0.1

x=np.linspace(-1,1,200)
noise=np.random.randn(x.size)/10
func=lambda a,b:np.sin(b*np.cos(a*x))
tf_func=lambda a,b:tf.sin(b*tf.cos(a*x))
y=func(*true_value)+noise
a=tf.Variable(initial_value[0],dtype=tf.float32)
b=tf.Variable(initial_value[1],dtype=tf.float32)
tf_y=tf_func(a,b)
mse=tf.reduce_mean(tf.square(tf_y-y))
train_op=tf.train.AdamOptimizer(LR).minimize(mse)
a_list,b_list,loss_list=[initial_value[0]],[initial_value[1]],[np.mean(np.square(func(*initial_value)-y))]
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(500):
		sess.run(train_op)
		a_,b_,loss_=sess.run([a,b,mse])
		a_list.append(a_)
		b_list.append(b_)
		loss_list.append(loss_)
print('a=%.4f,b=%.4f'%(a_list[-1],b_list[-1]))
result=func(a_list[-1],b_list[-1])
fig1=plt.figure(1)
ax=plt.subplot(111)
ax.plot(x,result,lw=5,c='blue')
ax.scatter(x,y)
plt.show()
fig2=plt.figure(2)
ax=Axes3D(fig2)
a_3d,b_3d=np.meshgrid(np.linspace(-1,6,100),np.linspace(-1,6,100))
loss_surface=np.array([np.mean(np.square(func(a_value,b_value)-y)) for a_value,b_value in zip(a_3d.flatten(),b_3d.flatten())]).reshape(a_3d.shape)
ax.scatter(a_list[0],b_list[0],loss_list[0],s=10,c='red')
ax.plot(a_list,b_list,np.array(loss_list),lw=3,c='r',zdir='z')
ax.plot_surface(a_3d,b_3d,loss_surface,cmap='rainbow',alpha=0.5)
ax.set_xlabel('a');ax.set_ylabel('b')
plt.show()

