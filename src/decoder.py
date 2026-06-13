import numpy as np
import sys
import os

# Resolve path issues
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .solution import Solution

class Decoder:
    """
    Decodes a Solution (Encoding 2) and calculates its robust fitness value.
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
        Main decoding function. Evaluates strategic investment costs and 
        calculates actual passenger operational transit times.
        """
        # Validate baseline feasibility (e.g., ensuring assigned hubs are actually open)
        if not solution.check_feasibility():
            solution.is_feasible = False
            solution.fitness = float('inf')
            return float('inf')
        
        # 1. Calculate First-Stage Investment Cost
        solution.total_invest_cost = np.dot(solution.Y, self.hub_cost) + solution.K * self.vehicle_cost
        
        # 2. Calculate Actual Operational Travel Time (Second-Stage)
        total_travel_time = 0.0
        unserved_passengers = 0
        
        # Track hub capacities (how many passengers use each hub)
        hub_usage = np.zeros(len(self.hub_cost))
        
        for p in range(len(solution.OriginHub)):
            orig_hub = solution.OriginHub[p]
            dest_hub = solution.DestHub[p]
            
            # Extract passenger walking components
            p_walk_to_orig = self.walk_time[p, orig_hub]
            p_walk_from_dest = self.walk_time[p, dest_hub]
            
            # Extract inter-hub transit distance/time from matrix
            inter_hub_transit = self.distance_matrix[orig_hub, dest_hub]
            
            # TODO: In the next phase, we will add the EV battery range check here.
            # If distance > battery_range, an intermediate recharge hub must be visited.
            
            # Accumulate operational time for this passenger
            passenger_time = p_walk_to_orig + inter_hub_transit + p_walk_from_dest
            total_travel_time += passenger_time
            
            # Update hub usage statistics for capacity constraints
            hub_usage[orig_hub] += 1
            hub_usage[dest_hub] += 1

        # 3. Handle Capacity & Fleet Constraints
        # Simple penalty rule: if total passengers exceed capacity boundaries, flag them
        max_allowed_passengers = solution.K * self.capacity
        if len(solution.OriginHub) > max_allowed_passengers:
            unserved_passengers = len(solution.OriginHub) - max_allowed_passengers

        # Assign calculated values back to the solution object
        solution.expected_time = total_travel_time
        solution.unserved_count = unserved_passengers
        solution.expected_penalty = solution.unserved_count * self.penalty_unserved
        
        # Final Objective Function Calculation
        solution.fitness = solution.total_invest_cost + solution.expected_time + solution.expected_penalty
        solution.is_feasible = True if unserved_passengers == 0 else False
        
        return solution.fitness