import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class TravelingSalesman:
    """Traveling Salesman Problem

    Parameters
    ----------
    cities : distance matrix
    stop_temp : terminal condition
    alpha : a positive constant
    internal : number of internal loops
    random_state : seed
        
    Returns
    -------
    distance_matrix, coordinates
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
        self.coords = np.column_stack((x, y))
              
        return np.int32(euclidean_distances(self.coords))
    
    
