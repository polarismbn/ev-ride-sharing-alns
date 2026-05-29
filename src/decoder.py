import numpy as np
import sys
import os

# import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .solution import Solution

class Decoder:
    """
    Decodes a Solution (Encoding 2) and calculates its fitness
    """
    
    def __init__(self, 
                 distance_matrix: np.ndarray,
                 walk_time: np.ndarray,
                 recharge_time: np.ndarray,
                 hub_cost: np.ndarray,
                 vehicle_cost: float,
                 capacity: int,
                 penalty_unserved: float):
        
        self.distance_matrix = distance_matrix
        self.walk_time = walk_time
        self.recharge_time = recharge_time
        self.hub_cost = hub_cost
        self.vehicle_cost = vehicle_cost
        self.capacity = capacity
        self.penalty_unserved = penalty_unserved

    def decode(self, solution: Solution) -> float:
        """
        Main decoding function - simple version for now
        """
        if not solution.check_feasibility():
            solution.is_feasible = False
            solution.fitness = float('inf')
            return float('inf')
        
        # Investment Cost
        solution.total_invest_cost = np.dot(solution.Y, self.hub_cost) + solution.K * self.vehicle_cost
        
        # Temporary estimation
        opened_hubs_count = np.sum(solution.Y)
        solution.expected_time = 1200 * opened_hubs_count + 80 * solution.K
        solution.unserved_count = 0
        solution.expected_penalty = solution.unserved_count * self.penalty_unserved
        
        solution.fitness = solution.total_invest_cost + solution.expected_time + solution.expected_penalty
        solution.is_feasible = True
        
        return solution.fitness