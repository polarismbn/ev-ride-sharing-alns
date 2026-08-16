"""
main.py
-------
Entry point for the EV Hub Location metaheuristic solver.

CHANGE IN THIS VERSION (feature-routing-timing):
    Added Horig(p) and Hdest(p) per passenger (Eqs 11-14 in paper).
    Each passenger now has a restricted set of allowed origin hubs
    and a restricted set of allowed destination hubs.
    This reflects geographic reality: a passenger can only board/alight
    at hubs within reasonable walking distance of their trip endpoints.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.decoder import Decoder
from src.solution import Solution
from src.genetic_algorithm import GeneticAlgorithm

# ══════════════════════════════════════════════════════════════
# 1. PROBLEM PARAMETERS
# ══════════════════════════════════════════════════════════════

NUM_HUBS = 20
V_MAX    = 20

np.random.seed(42)

# Nominal travel time t_ij between hubs (symmetric)
distance_matrix = np.random.randint(5, 45, size=(NUM_HUBS, NUM_HUBS)).astype(float)
np.fill_diagonal(distance_matrix, 0)
distance_matrix = (distance_matrix + distance_matrix.T) / 2

hub_cost         = np.random.randint(1000, 5000, NUM_HUBS).astype(float)
vehicle_cost     = 8000.0
penalty_unserved = 15000.0   # higher to account for routing inefficiency
BATTERY_RANGE    = 20.0
ATTACK_BUDGET    = 2
ATTACK_DELTA     = 8.0
LAMBDA_TIME      = 1.5
recharge_time    = np.random.randint(10, 25, NUM_HUBS).astype(float)

# ══════════════════════════════════════════════════════════════
# 2. PASSENGER DATA
#
# walk_time_full[p, h] = time for passenger p to walk to/from hub h
# Shape: (MAX_PASSENGERS, NUM_HUBS)
# ══════════════════════════════════════════════════════════════

MAX_PASSENGERS = 150
walk_time_full = np.random.randint(2, 10, (MAX_PASSENGERS, NUM_HUBS)).astype(float)

# ── Horig(p) and Hdest(p): allowed hubs per passenger (Eqs 11-14) ──────────
#
# Each passenger has:
#   Horig(p): set of hubs they can board from (closest hubs to trip origin)
#   Hdest(p): set of hubs they can alight at  (closest hubs to trip destination)
#
# We build these by sorting hubs by walk time for each passenger,
# then assigning the closest 1-3 hubs as Horig and the next 1-3 as Hdest.
# This ensures Horig and Hdest are disjoint (no boarding and alighting
# at the same hub) and reflect geographic proximity.

passenger_orig_hubs = []   # Horig(p) for each passenger p
passenger_dest_hubs = []   # Hdest(p) for each passenger p

rng_sets = np.random.default_rng(seed=99)   # separate seed for hub sets

for p in range(MAX_PASSENGERS):
    sorted_hubs = np.argsort(walk_time_full[p])   # closest hubs first

    # Horig(p): closest half of hubs as allowed origin hubs
    # Using half of hubs (not just 1-3) ensures passengers can always
    # find an open hub even when fewer hubs are open.
    # This makes the Pareto frontier more diverse.
    n_orig = max(2, NUM_HUBS // 2)
    orig   = sorted_hubs[:n_orig]

    # Hdest(p): remaining half as allowed destination hubs (disjoint from Horig)
    dest = sorted_hubs[n_orig:]

    passenger_orig_hubs.append(orig)
    passenger_dest_hubs.append(dest)

# ══════════════════════════════════════════════════════════════
# 3. DEMAND SCENARIOS
# ══════════════════════════════════════════════════════════════

scenarios = [
    {"name": "Normal",   "probability": 0.50, "n_passengers": 100, "omega": 1.0},
    {"name": "Moderate", "probability": 0.35, "n_passengers": 120, "omega": 1.3},
    {"name": "Heavy",    "probability": 0.15, "n_passengers": 150, "omega": 1.8},
]

NUM_PASSENGERS = max(s["n_passengers"] for s in scenarios)

print("Test data generated successfully.")
print(f"   Hubs={NUM_HUBS}, Max passengers={NUM_PASSENGERS}, V_max={V_MAX}")
print(f"   Penalty={penalty_unserved:,.0f}  (vehicle_cost={vehicle_cost:,.0f})")
print()
print("   Scenarios:")
for s in scenarios:
    print(f"   [{s['name']:8s}]  prob={s['probability']:.2f}  "
          f"passengers={s['n_passengers']:2d}  traffic x{s['omega']}")
print()
print("   Passenger hub sets (first 5 passengers):")
for p in range(5):
    print(f"   p={p}: Horig={list(passenger_orig_hubs[p])}  "
          f"Hdest={list(passenger_dest_hubs[p])}")

# ══════════════════════════════════════════════════════════════
# 4. INITIALIZE DECODER
# ══════════════════════════════════════════════════════════════

decoder = Decoder(
    distance_matrix      = distance_matrix,
    walk_time_full       = walk_time_full,
    recharge_time        = recharge_time,
    hub_cost             = hub_cost,
    vehicle_cost         = vehicle_cost,
    capacity             = 8,
    penalty_unserved     = penalty_unserved,
    battery_range        = BATTERY_RANGE,
    scenarios            = scenarios,
    attack_budget        = ATTACK_BUDGET,
    attack_delta         = ATTACK_DELTA,
    lambda_time          = LAMBDA_TIME,
    passenger_orig_hubs  = passenger_orig_hubs,
    passenger_dest_hubs  = passenger_dest_hubs,
)

print("\nDecoder initialized successfully.")

# ══════════════════════════════════════════════════════════════
# 5. MODE 1: SINGLE-OBJECTIVE GA
# ══════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  MODE 1: Single-Objective GA")
print("="*55)

ga_single = GeneticAlgorithm(
    decoder=decoder, num_hubs=NUM_HUBS, num_passengers=NUM_PASSENGERS,
    V_max=V_MAX, pop_size=60, n_generations=60,
    crossover_prob=0.9, mutation_prob=0.15, mutation_sigma=0.2,
    elite_ratio=0.1, pareto_mode=False, seed=42,
)

best = ga_single.evolve()

print("\n" + "-"*55)
print("  SINGLE-OBJ RESULT")
print("-"*55)
print(f"  Open hubs        : {list(best.opened_hubs)}")
print(f"  Active EVs (K)   : {best.K}")
print(f"  Investment cost  : {best.invest_cost:>12,.0f}")
print(f"  Expected time    : {best.expected_time:>12,.0f}")
print(f"  Unserved penalty : {best.expected_penalty:>12,.0f}")
print(f"  Total fitness    : {best.fitness:>12,.0f}")
print(f"  Feasible         : {best.is_feasible}")

# ══════════════════════════════════════════════════════════════
# 6. MODE 2: NSGA-II PARETO SEARCH
# ══════════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  MODE 2: NSGA-II Pareto Frontier")
print("="*55)

ga_pareto = GeneticAlgorithm(
    decoder=decoder, num_hubs=NUM_HUBS, num_passengers=NUM_PASSENGERS,
    V_max=V_MAX, pop_size=60, n_generations=60,
    crossover_prob=0.9, mutation_prob=0.15, mutation_sigma=0.2,
    elite_ratio=0.1, pareto_mode=True, seed=42,
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
    print(f"  {s.invest_cost:>12,.0f} | {recourse:>12,.0f} | "
          f"{s.count_open_hubs():>4} | {s.K:>4} | {str(s.is_feasible)}")

print(f"\n  Total Pareto points: {len(pareto)}")