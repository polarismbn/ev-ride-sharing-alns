"""
solution.py
-----------
Defines the chromosome structure for the EV Hub Location Problem.

CHROMOSOME LAYOUT (all values in [0, 1], length = 2*num_hubs + 1):

    Index 0 .. num_hubs-1       : hub_genes
        hub_genes[h] > 0.5  ->  hub h is opened  (yh = 1 in the paper)
        hub_genes[h] <= 0.5 ->  hub h is closed  (yh = 0)

    Index num_hubs              : vehicle_gene
        K = round(vehicle_gene * V_max)   number of active EVs

    Index num_hubs+1 .. end     : hub_priority
        Used by the greedy decoder to prefer certain hubs as
        intermediate recharge stops. Higher value = higher priority.

WHY REAL-VALUED CHROMOSOME?
    The paper uses binary variables (yh, zv). A real-valued chromosome
    in [0,1] is easier for crossover and mutation operators to work with,
    and the decoder converts it back to binary decisions.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Solution:
    """
    One candidate solution in the metaheuristic population.

    Fields set at construction (from chromosome):
        Y           : binary array, Y[h]=1 if hub h is open
        K           : number of active EVs

    Fields filled by Decoder.decode():
        invest_cost     : first-stage cost = sum(Ch*yh) + K*Cv
        expected_time   : expected passenger completion time across scenarios
        expected_penalty: worst-case unserved passenger penalty
        fitness         : scalar objective (invest_cost + lambda*time + penalty)
        is_feasible     : True if all passengers are served in every scenario

    Fields used by NSGA-II:
        rank            : Pareto front rank (0 = non-dominated)
        crowding_distance
    """

    chromosome:        np.ndarray
    num_hubs:          int
    V_max:             int

    # Decoded strategic decisions
    Y:                 np.ndarray = field(default=None, repr=False)
    K:                 int        = field(default=0)

    # Fitness fields (filled by decoder)
    invest_cost:       float = field(default=float('inf'))
    expected_time:     float = field(default=float('inf'))
    expected_penalty:  float = field(default=float('inf'))
    fitness:           float = field(default=float('inf'))
    is_feasible:       bool  = field(default=False)

    # NSGA-II fields
    rank:              int   = field(default=0)
    crowding_distance: float = field(default=0.0)

    def __post_init__(self):
        """Called automatically after dataclass __init__. Decodes Y and K."""
        self._decode_strategic()

    # ------------------------------------------------------------------
    # STRATEGIC DECODING
    # ------------------------------------------------------------------

    def _decode_strategic(self):
        """
        Decode hub opening decisions (Y) and EV count (K) from chromosome.

        Rules:
          - hub_genes[h] > 0.5  ->  open hub h
          - At least 2 hubs must be open (forced if chromosome gives fewer)
          - K = max(1, round(vehicle_gene * V_max))
        """
        hub_genes    = self.chromosome[:self.num_hubs]
        vehicle_gene = self.chromosome[self.num_hubs]

        self.Y = (hub_genes > 0.5).astype(int)

        # Force at least 1 hub open to avoid completely infeasible solutions.
        # We allow 1 hub (not 2) so the Pareto search can explore low-cost
        # configurations where only a few hubs are open.
        if int(np.sum(self.Y)) < 1:
            top1 = np.argsort(hub_genes)[-1:]
            self.Y[top1] = 1

        self.K = max(1, round(float(vehicle_gene) * self.V_max))

    # ------------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------------

    @property
    def hub_priority(self) -> np.ndarray:
        """
        Priority scores for each hub (last num_hubs genes in chromosome).
        Used by the decoder to rank intermediate recharge hubs.
        """
        return self.chromosome[self.num_hubs + 1:]

    @property
    def opened_hubs(self) -> np.ndarray:
        """Array of indices for hubs that are currently open."""
        return np.where(self.Y == 1)[0]

    def count_open_hubs(self) -> int:
        return int(np.sum(self.Y))

    # ------------------------------------------------------------------
    # FEASIBILITY CHECK
    # ------------------------------------------------------------------

    def is_structurally_feasible(self) -> bool:
        """
        Quick check before running the full decoder.
        Returns False if no hub is open or no EV is active.
        """
        return self.count_open_hubs() >= 1 and self.K >= 1

    # ------------------------------------------------------------------
    # FACTORY METHODS
    # ------------------------------------------------------------------

    @staticmethod
    def random(num_hubs: int, V_max: int, rng: np.random.Generator) -> "Solution":
        """Create a solution with a fully random chromosome in [0, 1]."""
        chrom = rng.random(2 * num_hubs + 1)
        return Solution(chromosome=chrom, num_hubs=num_hubs, V_max=V_max)

    @staticmethod
    def from_array(arr: np.ndarray, num_hubs: int, V_max: int) -> "Solution":
        """Create a solution from a pre-built array (clipped to [0, 1])."""
        return Solution(
            chromosome=np.clip(arr, 0.0, 1.0),
            num_hubs=num_hubs,
            V_max=V_max
        )

    def clone(self) -> "Solution":
        """
        Deep copy of this solution.
        Required before crossover/mutation to avoid aliasing bugs.
        """
        s = Solution(
            chromosome=self.chromosome.copy(),
            num_hubs=self.num_hubs,
            V_max=self.V_max
        )
        s.invest_cost      = self.invest_cost
        s.expected_time    = self.expected_time
        s.expected_penalty = self.expected_penalty
        s.fitness          = self.fitness
        s.is_feasible      = self.is_feasible
        return s

    # ------------------------------------------------------------------
    # PARETO DOMINANCE
    # ------------------------------------------------------------------

    def dominates(self, other: "Solution") -> bool:
        """
        Returns True if self Pareto-dominates other (minimization).

        self dominates other iff:
          - self is no worse than other in ALL objectives
          - self is strictly better in AT LEAST ONE objective

        Objectives used:
          f1 = invest_cost
          f2 = expected_time + expected_penalty
        """
        f_self  = (self.invest_cost,  self.expected_time  + self.expected_penalty)
        f_other = (other.invest_cost, other.expected_time + other.expected_penalty)
        better  = any(a < b for a, b in zip(f_self, f_other))
        no_worse = all(a <= b for a, b in zip(f_self, f_other))
        return better and no_worse

    # ------------------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Solution("
            f"open_hubs={list(self.opened_hubs)}, "
            f"K={self.K}, "
            f"invest={self.invest_cost:.0f}, "
            f"time={self.expected_time:.0f}, "
            f"penalty={self.expected_penalty:.0f}, "
            f"fitness={self.fitness:.0f}, "
            f"feasible={self.is_feasible})"
        )