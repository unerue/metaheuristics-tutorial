import os
import sys

# Add the parent directory for importing custom library
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import numpy as np
import matplotlib.pyplot as plt
from metaheuristics.problems import TravelingSalesman
from metaheuristics.neighbors import SimulatedAnnealing

import sys

if len(sys.argv) == 1:
    stop_temp = 0.00001
    alpha = 0.95
    max_iter = 200
else:   
    stop_temp = np.float64(sys.argv[1])
    alpha = np.float64(sys.argv[2])
    max_iter = np.int64(sys.argv[3])

cities = TravelingSalesman(100, 100, 20, 42).generate()
simulated_annealing = SimulatedAnnealing(cities, stop_temp, alpha, max_iter)
simulated_annealing.solve()


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

x = np.arange(len(simulated_annealing.history))
y = simulated_annealing.history

ln, = plt.plot([], [], lw=1, ls='-', c='red', label='Current values')

def init():
    ax.set_xlim(0, 10)
    ax.set_ylim([np.min(y)*0.995, np.max(y)*1.005])
    ax.grid(axis='y', linestyle='--')
    ax.set_title('Simulated Annealing')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness values')
    ax.legend(loc='upper right', frameon=True, shadow=False, 
              fancybox=False, ncol=1, framealpha=1, edgecolor='black')
    return ln, 

def update(i):
    ln.set_data(x[:i], y[:i])
    if i > 10:
        ax.set_xlim(0, np.max(x[:i])+5)
        # plt.axhline(np.min(y[:i]), c='g', lw=1, ls='--', label='Best fitness: {}'.format(np.min(y[:i])))
    return ln, 
  

ani = FuncAnimation(fig, update, init_func=init, interval=30, frames=np.arange(max_iter), repeat=True) # , blit=True
# ani = FuncAnimation(fig, update, interval=2, frames=ngen)

plt.show()
# To save the animation, use e.g.
#
# ani.save('simulated_annealing.gif', fps=60)