import numpy as np
import copy
from .solution import Solution

class GeneticAlgorithm:
    def __init__(self, decoder, num_hubs, num_passengers, pop_size=20, n_generations=30):
        self.decoder = decoder
        self.num_hubs = num_hubs
        self.num_passengers = num_passengers
        self.pop_size = pop_size
        self.n_generations = n_generations
        
        self.population = []
        self.best_sol = None

    def initialize_population(self):
        """Step 1: Generate initial feasible population randomly."""
        self.population = []
        for _ in range(self.pop_size):
            sol = Solution(self.num_hubs, self.num_passengers)
            
            # Ensure at least one hub is active (Binary vector Y)
            while np.sum(sol.Y) == 0:
                sol.Y = np.random.randint(0, 2, self.num_hubs)
            
            # Assign random initial fleet size K (e.g., between 5 and 30)
            sol.K = np.random.randint(5, 31)
            
            # Assign passengers only to opened hubs
            opened_hubs = np.where(sol.Y == 1)[0]
            for p in range(self.num_passengers):
                sol.OriginHub[p] = np.random.choice(opened_hubs)
                sol.DestHub[p] = np.random.choice(opened_hubs)
            
            # Evaluate fitness using the provided decoder
            self.decoder.decode(sol)
            self.population.append(sol)

    def evolve(self):
        """Step 2: Run the evolutionary loop to optimize the solution."""
        print("🚀 Starting Genetic Algorithm Optimization Loop...")
        self.initialize_population()
        
        for g in range(self.n_generations):
            # Sort population based on fitness (lower cost is better)
            self.population.sort(key=lambda x: x.fitness)
            
            # Update the global best solution found so far
            if self.best_sol is None or self.population[0].fitness < self.best_sol.fitness:
                self.best_sol = copy.deepcopy(self.population[0])
            
            print(f"📊 Generation [{g+1}/{self.n_generations}] -> Best Total Cost: {self.best_sol.fitness:,.0f}")
            
            # Create the next generation (Selection & Mutation)
            half_pop = self.pop_size // 2
            survivors = self.population[:half_pop]  # Select top 50% as elite parents
            
            new_generation = [copy.deepcopy(s) for s in survivors]
            
            while len(new_generation) < self.pop_size:
                # Randomly choose an elite parent
                parent = np.random.choice(survivors)
                child = copy.deepcopy(parent)
                
                # Apply Mutation: Randomly reassign origin/destination hub for a single passenger
                opened_hubs = np.where(child.Y == 1)[0]
                p_idx = np.random.randint(self.num_passengers)
                child.OriginHub[p_idx] = np.random.choice(opened_hubs)
                child.DestHub[p_idx] = np.random.choice(opened_hubs)
                
                # Re-evaluate child fitness
                self.decoder.decode(child)
                new_generation.append(child)
                
            self.population = new_generation
            
        return self.best_sol