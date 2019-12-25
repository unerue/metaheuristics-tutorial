import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory for importing custom library
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append('../')

from metaheuristics.problems import TravelingSalesman
from metaheuristics.population import SimpleEvolutionStrategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cities', type=int, default=20)
    parser.add_argument('-n', '--n_neighbors', type=int, default=20)
    parser.add_argument('--tabu_length', type=int, default=20)
    parser.add_argument('--max_iter', type=int, default=200)
    parser.add_argument('--savefig', type=bool, default=False)

    args = parser.parse_args()

    es = SimpleEvolutionStrategy(1, 2, 0, 3, 300)
    es.optimize()

    # Draw animiation 
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()

    x = np.arange(len(es.history['fitness']))
    y = es.history['fitness']

    ln, = plt.plot([], [], lw=1, c='red', ) # label='Best values'

    def init():
        ax.set_xlim(0, 3)
        ax.set_ylim([np.min(y)*0.995, np.max(y)*1.015])
        ax.grid(axis='y', linestyle='--')
        ax.set_title('(1+1)-ES $x^2\sin(x)^3$')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness values')
        # ax.legend(loc='upper right', frameon=True, shadow=False, 
        #           fancybox=False, ncol=1, framealpha=1, edgecolor='black')
        return ln, 

    def update(i):
        ln.set_data(x[:i], y[:i])
        # ln.set_data(x[:i], y[:i])
        if i > 10:
            ax.set_xlim(0, np.max(x[:i])+5)
            # plt.axhline(np.min(y[:i]), c='g', lw=1, ls='--', label='Best fitness: {}'.format(np.min(y[:i])))
        return ln, 
    
    ani = FuncAnimation(fig, update, init_func=init, interval=10, frames=np.arange(args.max_iter), repeat=False) # , blit=True
    plt.show()

    if args.savefig:
        # To save the animation, use e.g.
        ani.save('es-01.gif', fps=60)

    # New plot
    def f(x):
        return x**2 * np.sin(x)**3

    fig, ax = plt.subplots()

    x1 = np.linspace(0, 3, 100)
    y1 = f(x1)

    ln1, = plt.plot(x1, y1, lw=2, c='b', label='Optimal solution') # label='Best values'
    ln2, = plt.plot([], [], 'ro', alpha=0.4, ms=6, markeredgecolor='black', label='ES solution')


    x2 = es.history['solution']
    y2 = es.history['fitness']

    def init():
        ax.set_xlim(0, 3)
        ax.set_ylim([0, np.max(y1)*1.2])
        ax.grid(axis='y', linestyle='--')
        ax.set_title('(1+1)-ES $x^2\sin(x)^3$')
        ax.set_xlabel('Solution')
        ax.set_ylabel('Fitness values')
        ax.legend(loc='upper left', frameon=True, shadow=False, 
                  fancybox=False, ncol=1, framealpha=1, edgecolor='black')
        return ln1,  

    def update(i):
        ln2.set_data(x2[:i], y2[:i])
            # plt.axhline(np.min(y[:i]), c='g', lw=1, ls='--', label='Best fitness: {}'.format(np.min(y[:i])))
        return ln2, 
    
    ani = FuncAnimation(fig, update, init_func=init, interval=5, frames=np.arange(args.max_iter), repeat=False) # , blit=True
    plt.show()

    if args.savefig:
        # To save the animation, use e.g.
        ani.save('es-02.gif', fps=60)
    

if __name__ == '__main__':
    main()
