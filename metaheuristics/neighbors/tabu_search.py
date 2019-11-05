import numpy as np


class TabuSearch:
    """Tabu Search
    """
    def __init__(self, cities, n_neighbors, tabu_length, max_iter):
        self.cities = cities
        self.n_neighbors = n_neighbors
        self.tabu_length = tabu_length
        self.tabu_list = []
        self.max_iter = max_iter
        self.history = {'best': [], 'current': []}
    
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
                
        return new_solution, (i, j)

    def _get_neighbors(self, solution):
        candidate_list = []
        tabu_list = []

        n = 0
        while n < self.n_neighbors:
            candidate, tabu = self._two_opt(solution)
            if candidate not in candidate_list:
                candidate_list.append(candidate)
                tabu_list.append(tabu)
                n += 1
                
        return candidate_list, tabu_list
    
    def _eval_aspiration(self, candidate_list, tabu_list):
        # Remove candidate
        for i, candidate in enumerate(candidate_list):
            value = self._fitness(candidate)
            if tabu_list[i] in self.tabu_list:
                if value < self.aspiration_level:
                    pass
                else:
                    del candidate_list[i]
                    del tabu_list[i]
        
        # Evalute each candidate
        current_value = np.inf
        for candidate, tabu in zip(candidate_list, tabu_list):
            value = self._fitness(candidate)
            if value < current_value:
                current_value = value
                current_solution = candidate
                current_tabu = tabu
        
        # Update tabu list
        if len(self.tabu_list) < self.tabu_length:
            self.tabu_list.append(current_tabu)
        else:
            self.tabu_list.pop(0)
            self.tabu_list.append(current_tabu)
        
        return current_solution

    def solve(self):
        # Initial solution
        initial_solution = list(np.random.permutation(range(1, len(self.cities))))
        initial_solution.insert(0, 0)
        initial_value = self._fitness(initial_solution)
        
        current_solution = initial_solution
        current_value = initial_value
        self.aspiration_level = initial_value
        
        # Initialize best value
        best_value = np.inf
        best_solution = None
        while self.max_iter:
            # Generate candidates
            candidate_list, tabu_list = self._get_neighbors(current_solution)
            
            # Evaluating tabu and aspiration
            current_solution = self._eval_aspiration(candidate_list, tabu_list)
            current_value = self._fitness(current_solution)
            
            if current_value < best_value:
                best_value = current_value
                best_solution = current_solution
                if best_value < self.aspiration_level:
                    self.aspiration_level = best_value
            
            self.history['best'].append(best_value)
            self.history['current'].append(current_value)

            self.max_iter -= 1
        
        return best_value


if __name__ == '__main__':
    pass