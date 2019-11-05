import numpy as np


class SimulatedAnnealing:
    """Simulated Annealing
    """
    def __init__(self, cities, stop_temp, alpha, max_iter):
        self.cities = cities
        self.stop_temp = stop_temp
        self.alpha = alpha
        self.max_iter = max_iter  # internal loops
        self.history = []

    def _fitness(self, solution):
        return np.sum([self.cities[i, j] for i, j in zip(solution, solution[1:] + [solution[0]])])
        
    def _two_opt(self, solution):
        new_solution = solution.copy()
        changed = True
        while changed:
            i = np.random.choice(range(1, len(solution)-2))
            j = np.random.choice(range(i+1, len(solution)))
            if j-i != 1:
                new_solution[i:j] = reversed(solution[i:j])
                changed = False
            else:
                continue
                
        return new_solution

    def _acceptance_probability(self, delta):
        return np.exp(-delta / self.temp)

    def solve(self):
        # Initial solution
        initial_solution = list(np.random.permutation(range(1, len(self.cities))))
        initial_solution.insert(0, 0)
        self.initial_value = self._fitness(initial_solution)

        current_solution = initial_solution
        current_value = self.initial_value
               
        # Initial temperature
        self.temp = self.initial_value
        while self.temp >= self.stop_temp:
            for i in range(self.max_iter):
                candidate_solution = self._two_opt(current_solution)
                candidate_value = self._fitness(candidate_solution)
                
                delta = candidate_value - current_value
                if delta <= 0:
                    current_solution = candidate_solution
                    current_value = candidate_value
                                      
                elif self._acceptance_probability(delta) >= np.random.random():
                    current_solution = candidate_solution
                    current_value = candidate_value

            self.temp = self.alpha * self.temp
            self.history.append(current_value)
        return current_value