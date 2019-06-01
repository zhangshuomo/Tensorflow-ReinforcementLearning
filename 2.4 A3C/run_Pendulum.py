from A3C import A3C_Worker
import threading as td
import matplotlib.pyplot as plt
import multiprocessing as mp


n_threadings=mp.cpu_count()
workers=[]
threadings=[]

for i in range(n_threadings):
	worker=A3C_Worker('w{0}'.format(i+1), 'Pendulum-v0')
	workers.append(worker)
	t=td.Thread(target=worker.work)
	t.start()
	threadings.append(t)
	
[t.join() for t in threadings]

plt.plot(A3C_Worker.moving_rewards())
plt.show()
