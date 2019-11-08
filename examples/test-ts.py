import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from metaheuristics.problems import TravelingSalesman
from metaheuristics.neighbors import TabuSearch

# Add the parent directory for importing custom library
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

if len(sys.argv) == 1:
    n_neighbors = 20
    tabu_length = 20
    max_iter = 200
else:   
    n_neighbors = int(sys.argv[1])
    tabu_length = int(sys.argv[2])
    max_iter = int(sys.argv[3])

cities = TravelingSalesman(100, 100, 30, 42).generate()
tabu_search = TabuSearch(cities, n_neighbors, tabu_length, max_iter)
tabu_search.solve()


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

x = np.arange(len(tabu_search.history['best']))
y1 = tabu_search.history['current']
y2 = tabu_search.history['best']

ln1, = plt.plot([], [], lw=0.7, ls='--', c='blue', label='Current values')
ln2, = plt.plot([], [], lw=1, c='red', label='Best values')

def init():
    ax.set_xlim(0, 10)
    ax.set_ylim([np.min(y1)*0.995, np.max(y1)*1.005])
    ax.grid(axis='y', linestyle='--')
    ax.set_title('Tabu Search')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Fitness values')
    ax.legend(loc='upper right', frameon=True, shadow=False, 
              fancybox=False, ncol=1, framealpha=1, edgecolor='black')
    return ln1, 

def update(i):
    ln1.set_data(x[:i], y1[:i])
    ln2.set_data(x[:i], y2[:i])
    if i > 10:
        ax.set_xlim(0, np.max(x[:i])+5)
        # plt.axhline(np.min(y[:i]), c='g', lw=1, ls='--', label='Best fitness: {}'.format(np.min(y[:i])))
    return ln1, ln2
  
ani = FuncAnimation(fig, update, init_func=init, interval=30, frames=np.arange(max_iter), repeat=True) # , blit=True
plt.show()

# To save the animation, use e.g.
ani.save('tabu_search.gif', fps=60)