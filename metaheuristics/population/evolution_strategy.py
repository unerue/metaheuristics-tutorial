import numpy as np


class SimpleEvolutionStrategy:
    """ (1+1)-ES
    """
    def __init__(self, n_vars, sigma, lower_bound, upper_bound, max_iter):
        self.n_vars = n_vars
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.history = {'fitness': [], 'solution': []}
    
    def _check_range(self, x):
        if self.lower_bound <= x <= self.upper_bound:
            return True
    
    def _fitness(self, x):
        return x**2 * np.sin(x)**3
    
    def optimize(self):
        # x = np.random.randint(self.lower_bound, self.upper_bound, size=self.n_vars)
        # x = 0
        x = np.random.rand()
        while self.max_iter:
            o = x + np.random.normal(0, self.sigma, size=self.n_vars)
            if self._check_range(o):
                if self._fitness(x) < self._fitness(o):
                    x = o
                else:
                    x = x
            else:
                pass
                
            self.history['fitness'].append(self._fitness(x))
            self.history['solution'].append(x)
            self.max_iter -= 1