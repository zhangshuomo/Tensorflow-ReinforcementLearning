import matplotlib.pyplot as plt
from PPO_Worker import PPO_Worker
import threading as td
import multiprocessing as mp

n_threads=mp.cpu_count()
threads=[]
for t in range(n_threads):
	agent=PPO_Worker('CartPole-v0','w{}'.format(t+1))
	thread=td.Thread(target=agent.work)
	threads.append(thread)
thread=td.Thread(target=PPO_Worker.updateParameters)
threads.append(thread)
[t.start() for t in threads]
[t.join() for t in threads]
plt.figure()
plt.plot(PPO_Worker.get_moving_rewards())
plt.show()

