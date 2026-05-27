import numpy as np
from typing import List, Optional

class Solution:
    """
    Represents a solution for Encoding 2 (Simple Version)
    - Hub opening decisions
    - Number of vehicles
    - Passenger to hub assignments
    """
    
    def __init__(self, num_hubs: int, num_passengers: int):
        # First-stage decisions
        self.Y = np.zeros(num_hubs, dtype=int)           # Opened hubs (binary vector)
        self.K = 0                                       # Number of active vehicles
        
        # Passenger assignment
        self.OriginHub = np.zeros(num_passengers, dtype=int)
        self.DestHub = np.zeros(num_passengers, dtype=int)
        
        # Evaluation values (filled after decoding)
        self.fitness: float = float('inf')
        self.total_invest_cost: float = 0.0
        self.expected_time: float = 0.0
        self.expected_penalty: float = 0.0
        self.unserved_count: int = 0
        self.is_feasible: bool = False
        
        # To store routes after decoding (optional)
        self.routes: Optional[List] = None

    def is_feasible(self) -> bool:
        """Check if solution is feasible"""
        opened_hubs = set(np.where(self.Y == 1)[0])
        
        for p in range(len(self.OriginHub)):
            if (self.OriginHub[p] not in opened_hubs or 
                self.DestHub[p] not in opened_hubs):
                return False
                
        if self.K < 1:
            return False
            
        return True