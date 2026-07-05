"""
main.py
-------
Entry point for the EV Hub Location metaheuristic solver.

WHAT THIS FILE DOES:
    1. Generates synthetic test data matching the paper's structure
    2. Defines three demand scenarios (Normal, Moderate, Heavy)
    3. Initializes the Decoder with all problem parameters
    4. Runs single-objective GA (Mode 1) and NSGA-II Pareto search (Mode 2)
    5. Prints results

SCENARIO DESIGN:
    Each scenario has three components:
        probability  (ps)       : weight in the ExpectedTime objective (Eq. 4)
        n_passengers            : active passengers in this scenario (ds,p in paper)
        omega                   : traffic congestion multiplier

    Using scenario-specific passenger counts (not just traffic multipliers)
    is closer to the paper, which defines different demand ds,p per scenario.

PENALTY CALIBRATION:
    penalty_unserved must exceed vehicle_cost to force the algorithm to
    buy enough EVs for all scenarios, including the worst-case (Heavy).
    With vehicle_cost=8000: penalty_unserved=9000 works correctly.
    See decoder.py module docstring for detailed explanation.

HOW TO RUN:
    python main.py

    Make sure the folder structure is:
        ev-ride-sharing/
            main.py
            src/
                __init__.py
                solution.py
                decoder.py
                genetic_algorithm.py
"""

import numpy as np
import sys
import os

# Add project root to path so 'src' is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.decoder import Decoder
from src.solution import Solution
from src.genetic_algorithm import GeneticAlgorithm

# ══════════════════════════════════════════════════════════════
# 1. PROBLEM PARAMETERS
# ══════════════════════════════════════════════════════════════

NUM_HUBS = 8      # |H|: number of candidate hub locations
V_MAX    = 10     # maximum number of EVs that can be activated

np.random.seed(42)   # fixed seed for reproducible results

# Nominal travel time t_ij between hubs (symmetric matrix)
distance_matrix = np.random.randint(5, 45, size=(NUM_HUBS, NUM_HUBS)).astype(float)
np.fill_diagonal(distance_matrix, 0)
distance_matrix = (distance_matrix + distance_matrix.T) / 2   # make symmetric

# Fixed hub opening cost C_h for each hub
hub_cost = np.random.randint(1000, 5000, NUM_HUBS).astype(float)

# Fixed EV activation cost C_v (identical for all vehicles, as in the paper)
vehicle_cost = 8000.0

# Penalty per unserved passenger C_{p,s}
# Must be > vehicle_cost (8000) to ensure feasibility is enforced.
# With 9000: leaving 1 passenger unserved costs more than activating 1 extra EV.
penalty_unserved = 9000.0

# Maximum EV travel distance on a full battery charge
BATTERY_RANGE = 20.0

# Adversarial attack parameters
# Gamma: max number of arcs the adversary can slow down
# Delta: extra travel time added to each attacked arc
ATTACK_BUDGET = 2
ATTACK_DELTA  = 8.0

# Lambda: monetary value of one unit of travel time (converts time to cost)
LAMBDA_TIME = 1.5

# Recharge time r_h at each hub (deterministic, as in the paper)
recharge_time = np.random.randint(10, 25, NUM_HUBS).astype(float)

# ══════════════════════════════════════════════════════════════
# 2. DEMAND SCENARIOS
#
# The paper defines ds,p: demand of passenger p in scenario s.
# We model this as n_passengers active passengers per scenario,
# with a shared walk_time matrix for all possible passengers.
# ══════════════════════════════════════════════════════════════

# Maximum passengers across all scenarios (worst case = Heavy scenario)
MAX_PASSENGERS = 65

# Walk time w_{p,h}: time for passenger p to walk to/from hub h
# Shape: (MAX_PASSENGERS, NUM_HUBS) -- covers all possible passengers
walk_time_full = np.random.randint(2, 10, (MAX_PASSENGERS, NUM_HUBS)).astype(float)

# Three demand scenarios matching paper structure
scenarios = [
    {
        "name":         "Normal",
        "probability":  0.50,    # ps = 0.50
        "n_passengers": 40,      # 40 active passengers
        "omega":        1.0,     # nominal traffic
    },
    {
        "name":         "Moderate",
        "probability":  0.35,    # ps = 0.35
        "n_passengers": 50,      # 50 active passengers
        "omega":        1.3,     # 30% slower travel times
    },
    {
        "name":         "Heavy",
        "probability":  0.15,    # ps = 0.15
        "n_passengers": 65,      # 65 active passengers (worst case)
        "omega":        1.8,     # 80% slower travel times
    },
]

# Number of passengers in worst-case scenario (used for chromosome sizing)
NUM_PASSENGERS = max(s["n_passengers"] for s in scenarios)

print("Test data generated successfully.")
print(f"   Hubs        : {NUM_HUBS}")
print(f"   Max passengers (worst scenario): {NUM_PASSENGERS}")
print(f"   V_max       : {V_MAX}")
print(f"   Penalty     : {penalty_unserved:,.0f}  (vehicle_cost={vehicle_cost:,.0f})")
print()
print("   Scenarios:")
for s in scenarios:
    print(
        f"   [{s['name']:8s}]  "
        f"prob={s['probability']:.2f}  "
        f"passengers={s['n_passengers']:2d}  "
        f"traffic x{s['omega']}"
    )

# ══════════════════════════════════════════════════════════════
# 3. INITIALIZE DECODER
# ══════════════════════════════════════════════════════════════

decoder = Decoder(
    distance_matrix  = distance_matrix,
    walk_time_full   = walk_time_full,
    recharge_time    = recharge_time,
    hub_cost         = hub_cost,
    vehicle_cost     = vehicle_cost,
    capacity         = 8,              # Q: seats per EV
    penalty_unserved = penalty_unserved,
    battery_range    = BATTERY_RANGE,
    scenarios        = scenarios,
    attack_budget    = ATTACK_BUDGET,
    attack_delta     = ATTACK_DELTA,
    lambda_time      = LAMBDA_TIME,
)

print("\nDecoder initialized successfully.")

# ══════════════════════════════════════════════════════════════
# 4. MODE 1: SINGLE-OBJECTIVE GA
#    Minimizes: InvestCost + lambda*ExpectedTime + WorstPenalty
# ══════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  MODE 1: Single-Objective GA")
print("="*55)

ga_single = GeneticAlgorithm(
    decoder        = decoder,
    num_hubs       = NUM_HUBS,
    num_passengers = NUM_PASSENGERS,
    V_max          = V_MAX,
    pop_size       = 60,
    n_generations  = 50,
    crossover_prob = 0.9,
    mutation_prob  = 0.15,
    mutation_sigma = 0.2,
    elite_ratio    = 0.1,
    pareto_mode    = False,
    seed           = 42,
)

best = ga_single.evolve()

print("\n" + "-"*55)
print("  SINGLE-OBJ RESULT")
print("-"*55)
print(f"  Open hubs           : {list(best.opened_hubs)}")
print(f"  Active EVs (K)      : {best.K}")
print(f"  Investment cost     : {best.invest_cost:>12,.0f}")
print(f"  Expected time       : {best.expected_time:>12,.0f}")
print(f"  Unserved penalty    : {best.expected_penalty:>12,.0f}")
print(f"  Total fitness       : {best.fitness:>12,.0f}")
print(f"  Feasible            : {best.is_feasible}")

# ══════════════════════════════════════════════════════════════
# 5. MODE 2: NSGA-II BI-OBJECTIVE PARETO SEARCH
#    Objective 1: InvestCost
#    Objective 2: ExpectedTime + WorstPenalty (recourse cost)
# ══════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  MODE 2: NSGA-II Pareto Frontier")
print("="*55)

ga_pareto = GeneticAlgorithm(
    decoder        = decoder,
    num_hubs       = NUM_HUBS,
    num_passengers = NUM_PASSENGERS,
    V_max          = V_MAX,
    pop_size       = 60,
    n_generations  = 50,
    crossover_prob = 0.9,
    mutation_prob  = 0.15,
    mutation_sigma = 0.2,
    elite_ratio    = 0.1,
    pareto_mode    = True,
    seed           = 42,
)

ga_pareto.evolve()

print("\n" + "-"*55)
print("  PARETO FRONT SOLUTIONS")
print("-"*55)
print(f"  {'InvestCost':>12} | {'Recourse':>12} | {'Hubs':>4} | {'EVs':>4} | Feasible")
print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*4}-+-{'-'*4}-+-{'-'*8}")

pareto = sorted(ga_pareto.pareto_front, key=lambda s: s.invest_cost)
for s in pareto:
    recourse = s.expected_time + s.expected_penalty
    print(
        f"  {s.invest_cost:>12,.0f} | {recourse:>12,.0f} | "
        f"{s.count_open_hubs():>4} | {s.K:>4} | {str(s.is_feasible)}"
    )

print(f"\n  Total Pareto points: {len(pareto)}")