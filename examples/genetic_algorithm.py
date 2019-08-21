import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import sys


num_cities = int(sys.argv[1])
pop_size = int(sys.argv[2])
num_iters = int(sys.argv[3])


class GenerateCities:
    """Generate Cities
    
    Generating coordinates and distance matrix randomly.
    
    Parameters
    ----------
    x : x coordinate of a city
    y : y coordinate of a city
    num_cities: number of cities
    random_state : seed
    
    Returns
    -------
    coords : city coordinates 
    matrix : distance numpy matrix
    """
    def __init__(self, x, y, num_cities, random_state=None):
        self.x = x
        self.y = y
        self.num_cities = num_cities
        self.random_state = random_state

    def generate(self):
        np.random.seed(self.random_state)
        x = np.random.randint(self.x, size=self.num_cities)
        y = np.random.randint(self.y, size=self.num_cities)
        coords = np.column_stack((x, y))
              
        return coords, np.int32(euclidean_distances(coords))


coords, cities = GenerateCities(100, 100, num_cities, 42).generate()

# from pulp import *

# # Initialize travelling salesman problem
# prob = LpProblem('Travelling Salesman', LpMinimize)

# n = len(cities)
# indexs = [(i, j) for i in range(n) for j in range(n) if i != j]

# # Creating decision variables
# x = LpVariable.dicts('x', indexs, cat='Binary')
# u = LpVariable.dicts('u', list(range(n)), lowBound=0, upBound=n-1, cat='Continuous')

# # Objective function
# prob += lpSum([cities[i][j] * x[(i,j)] for i, j in indexs])

# # Constraints
# for i in range(n):
#     prob += lpSum([x[(i,j)] for j in range(n) if i != j]) == 1
    
# for j in range(n):
#     prob += lpSum([x[(i,j)] for i in range(n) if i != j]) == 1
    
# for i in range(1, n):
#     for j in range(1, n):
#         if i != j:
#             prob += u[i] - u[j] + n * x[(i,j)] <= n - 1

# # Solve problem
# prob.solve()
# print(value(prob.objective))


import random
import array
from deap import base
from deap import creator
from deap import tools
from deap import algorithms


creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register('attr_cities', random.sample, range(1,len(cities)), len(cities)-1)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_cities)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


def eval_distance(individual):
    path = [0] + individual
    return np.sum([cities[i,j] for i, j in zip(path, path[1:] + [path[0]])]),

def two_point(ind):
    opt = True
    while opt:
        i = random.choice(range(1,len(cities)-2))
        j = random.choice(range(i,len(cities)))
        if i < j:
            ind[i:j] = reversed(ind[i:j])
            opt = False
    return ind

def cxTwoOpt(ind1, ind2):
    ind1 = two_point(ind1)
    ind2 = two_point(ind2)
    return ind1, ind2


toolbox.register('evaluate', eval_distance)
toolbox.register('mate', cxTwoOpt)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=0.05)
toolbox.register('select', tools.selTournament, tournsize=3)
 

def main():
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=num_iters, stats=stats, halloffame=hof, verbose=True)
    return pop, log

_, log = main()

print('num_cities: {}, pop_size: {}, num_gen: {}'.format(num_cities, pop_size, num_iters))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(dpi=100)

iters = list(range(len(log)))
min_vals = [log[i]['min'] for i in range(len(log))]
avg_vals = [log[i]['avg'] for i in range(len(log))]
std_vals = [log[i]['max'] for i in range(len(log))]

ax.scatter(iters, min_vals, marker='.', label='min')
ax.scatter(iters, avg_vals, marker='.', label='avg')
ax.scatter(iters, std_vals, marker='.', label='max')
ax.legend()
ax.grid(axis='y', linestyle='--')
plt.show()


