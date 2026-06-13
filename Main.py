import numpy as np
import sys
import os

# Resolve path issues to ensure src folder is discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.decoder import Decoder
from src.solution import Solution
from src.genetic_algorithm import GeneticAlgorithm

# ====================== 1. Synthetic Test Data Generation ======================
num_hubs = 8
num_passengers = 50

# Set seed for reproducibility across runs
np.random.seed(42)

# Generate randomized distance matrix between candidate hubs
distance_matrix = np.random.randint(5, 45, size=(num_hubs, num_hubs))
np.fill_diagonal(distance_matrix, 0)

# Strategic costs and operational boundaries
hub_cost = np.random.randint(1000, 5000, num_hubs).astype(float)
vehicle_cost = 8000.0
capacity = 8
penalty_unserved = 1000.0

# Passenger-specific metrics (walk times to hubs and recharge overheads)
walk_time = np.random.randint(2, 10, (num_passengers, num_hubs))
recharge_time = np.random.randint(10, 25, num_hubs)

print(" Synthetic test data successfully prepared.\n")

# ====================== 2. Initialize Operational Decoder ======================
# UPDATE: The battery_range parameter is added here to support EV constraints
decoder = Decoder(
    distance_matrix=distance_matrix,
    walk_time=walk_time,
    recharge_time=recharge_time,
    hub_cost=hub_cost,
    vehicle_cost=vehicle_cost,
    capacity=capacity,
    penalty_unserved=penalty_unserved,
    battery_range=20.0  # Added to activate EV battery-constrained routing logic
)

print("⚙️ Laboratory decoder initialized successfully. Proceeding to GA execution...")

# ====================== 3. Evolutionary Optimization Loop ======================
# Initialize the Genetic Algorithm and inject the prepared decoder
ga = GeneticAlgorithm(
    decoder=decoder,
    num_hubs=num_hubs,
    num_passengers=num_passengers,
    pop_size=20,         # Total population size per generation
    n_generations=15     # Number of evolutionary generations for rapid testing
)

# Run the evolutionary optimization process
best_solution = ga.evolve()

# ====================== 4. Display Final Optimal Results ======================
print("\n === Final Optimization Result ===")
print(best_solution)
print(f" Final Investment Cost: {best_solution.total_invest_cost:,.0f}")
print(f" Final Expected Travel Time: {best_solution.expected_time:,.0f}")
print(f" Unserved Passengers Count: {best_solution.unserved_count}")