"""
decoder.py
----------
Greedy Decoder — fast replacement for the exact MILP in the paper.

CHANGES IN THIS VERSION:
    Integrated per-EV routing via Router class.
    Instead of a batch capacity check, each EV now gets a proper route
    with specific passengers assigned to it.

APPROXIMATIONS vs THE PAPER:
    Paper component                      Our approximation
    ────────────────────────────────     ─────────────────────────────────────
    Recourse MILP (Eqs 15-36)           Greedy Router (router.py)
    CCG attacker subproblem             Two-phase local search
    Tin/Tout timing (Eqs 30-33)         Sequential schedule in Router
    Horig(p), Hdest(p) (Eqs 11-14)     Implemented
    Per-arc capacity (Eq 17)            Enforced per EV route
    Flow conservation (Eq 15)          Enforced by construction in Router
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from .solution import Solution
from .router import Router


class Decoder:
    """
    Evaluates a Solution and fills its fitness fields.

    Parameters
    ----------
    distance_matrix      : (H, H) float  — nominal travel time t_ij
    walk_time_full       : (P_max, H) float — walk time w_{p,h}
    recharge_time        : (H,) float    — recharge delay r_h
    hub_cost             : (H,) float    — opening cost C_h
    vehicle_cost         : float         — activation cost C_v per EV
    capacity             : int           — seats per EV (Q)
    penalty_unserved     : float         — penalty per unserved passenger
    battery_range        : float         — max EV range on full charge
    scenarios            : list of dict  — probability, n_passengers, omega
    attack_budget        : int           — Gamma
    attack_delta         : float         — Delta
    lambda_time          : float         — lambda
    passenger_orig_hubs  : Horig(p) per passenger
    passenger_dest_hubs  : Hdest(p) per passenger
    """

    def __init__(
        self,
        distance_matrix:     np.ndarray,
        walk_time_full:      np.ndarray,
        recharge_time:       np.ndarray,
        hub_cost:            np.ndarray,
        vehicle_cost:        float,
        capacity:            int,
        penalty_unserved:    float,
        battery_range:       float            = 25.0,
        scenarios:           List[Dict]       = None,
        attack_budget:       int              = 0,
        attack_delta:        float            = 5.0,
        lambda_time:         float            = 1.0,
        passenger_orig_hubs: Optional[List]   = None,
        passenger_dest_hubs: Optional[List]   = None,
    ):
        self.distance_matrix     = distance_matrix.astype(float)
        self.walk_time_full      = walk_time_full.astype(float)
        self.recharge_time       = recharge_time.astype(float)
        self.hub_cost            = hub_cost.astype(float)
        self.vehicle_cost        = float(vehicle_cost)
        self.capacity            = int(capacity)
        self.penalty_unserved    = float(penalty_unserved)
        self.battery_range       = float(battery_range)
        self.attack_budget       = int(attack_budget)
        self.attack_delta        = float(attack_delta)
        self.lambda_time         = float(lambda_time)
        self.num_hubs            = distance_matrix.shape[0]
        self.passenger_orig_hubs = passenger_orig_hubs
        self.passenger_dest_hubs = passenger_dest_hubs

        if scenarios is None:
            n_p = walk_time_full.shape[0]
            self.scenarios = [{"probability": 1.0, "n_passengers": n_p, "omega": 1.0}]
        else:
            self.scenarios = scenarios

        self.max_passengers = max(s["n_passengers"] for s in self.scenarios)

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------

    def decode(self, solution: Solution) -> float:
        """
        Evaluate a solution and return scalar fitness.

        Steps:
          1. Structural check
          2. Investment cost (Eq. 3)
          3. For each scenario:
               a. Assign passengers to allowed hubs (Horig/Hdest)
               b. Apply adversarial attack
               c. Build per-EV routes with Router
               d. Collect recourse time and penalty
          4. ExpectedTime = sum_s(ps * rec_time_s)     (Eq. 4)
             WorstPenalty = max_s(rec_penalty_s)
          5. fitness = InvestCost + lambda*ExpectedTime + WorstPenalty
        """
        if not solution.is_structurally_feasible():
            solution.is_feasible = False
            solution.fitness     = float('inf')
            return float('inf')

        opened = solution.opened_hubs

        # First-stage investment cost (Eq. 3)
        solution.invest_cost = (
            float(np.dot(solution.Y, self.hub_cost))
            + solution.K * self.vehicle_cost
        )

        expected_time = 0.0
        worst_penalty = 0.0
        all_feasible  = True

        for s in self.scenarios:
            ps    = s["probability"]
            n_p   = s["n_passengers"]
            omega = s["omega"]

            walk_s = self.walk_time_full[:n_p, :]

            # Step a: assign passengers to hubs (Horig/Hdest)
            origin_hubs, dest_hubs, n_unassigned = self._assign_passengers(
                opened, walk_s, n_p
            )

            # Step b: adversarial attack
            attacked_dist = self._apply_attack(
                self.distance_matrix, omega, solution
            )

            # Step c: build per-EV routes
            router = Router(
                distance_matrix     = attacked_dist,
                walk_time           = walk_s,
                recharge_time       = self.recharge_time,
                battery_range       = self.battery_range,
                capacity            = self.capacity,
                passenger_orig_hubs = self.passenger_orig_hubs,
                passenger_dest_hubs = self.passenger_dest_hubs,
                omega               = omega,
            )

            routes, n_unserved_routing, rec_time = router.build_routes(
                opened, solution.K, origin_hubs, dest_hubs
            )

            # Step d: penalty includes unassigned + unserved by routing
            total_unserved = n_unassigned + n_unserved_routing
            rec_penalty    = total_unserved * self.penalty_unserved

            # Probability-weighted time (Eq. 4)
            expected_time += ps * rec_time

            # Worst-case penalty
            if rec_penalty > worst_penalty:
                worst_penalty = rec_penalty

            if rec_penalty > 0:
                all_feasible = False

        solution.expected_time    = expected_time
        solution.expected_penalty = worst_penalty
        solution.is_feasible      = all_feasible

        # TotalMonetary (Eq. 7)
        solution.fitness = (
            solution.invest_cost
            + self.lambda_time * solution.expected_time
            + solution.expected_penalty
        )

        return solution.fitness

    # ------------------------------------------------------------------
    # PASSENGER ASSIGNMENT  (Eqs 11-14)
    # ------------------------------------------------------------------

    def _assign_passengers(
        self,
        opened: np.ndarray,
        walk_s: np.ndarray,
        n_p:    int,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Assign each passenger to their best available open hub
        within Horig(p) and Hdest(p).

        Returns origin_hubs, dest_hubs (both -1 if unassignable),
        and n_unassigned count.
        """
        origin_hubs  = np.full(n_p, -1, dtype=int)
        dest_hubs    = np.full(n_p, -1, dtype=int)
        n_unassigned = 0

        opened_set = set(opened.tolist())

        for p in range(n_p):
            if self.passenger_orig_hubs is not None:
                allowed_orig = [h for h in self.passenger_orig_hubs[p]
                                if h in opened_set]
            else:
                allowed_orig = list(opened)

            if not allowed_orig:
                n_unassigned += 1
                continue

            orig = min(allowed_orig, key=lambda h: walk_s[p, h])

            if self.passenger_dest_hubs is not None:
                allowed_dest = [h for h in self.passenger_dest_hubs[p]
                                if h in opened_set]
            else:
                allowed_dest = [h for h in opened if h != orig]

            if not allowed_dest:
                n_unassigned += 1
                continue

            dest = min(allowed_dest, key=lambda h: walk_s[p, h])

            origin_hubs[p] = orig
            dest_hubs[p]   = dest

        return origin_hubs, dest_hubs, n_unassigned

    # ------------------------------------------------------------------
    # ADVERSARIAL ATTACK  (two-phase local search)
    # ------------------------------------------------------------------

    def _apply_attack(
        self,
        dist:     np.ndarray,
        omega:    float,
        solution: "Solution" = None,
    ) -> np.ndarray:
        """
        Two-phase adversarial attack.
        Phase 1: score-based initialisation (prefer arcs between open hubs).
        Phase 2: local search swap to find worse attacks.
        """
        base = dist * omega

        if self.attack_budget <= 0:
            return base

        H        = self.num_hubs
        all_arcs = [(i, j) for i in range(H) for j in range(H) if i != j]

        arc_scores = {}
        for (i, j) in all_arcs:
            score = base[i, j]
            if solution is not None:
                opened = solution.opened_hubs
                if i in opened and j in opened:
                    score *= 1.5
            arc_scores[(i, j)] = score

        sorted_arcs  = sorted(all_arcs, key=lambda a: arc_scores[a], reverse=True)
        attacked_set = set(sorted_arcs[:self.attack_budget])

        def build_matrix(aset):
            m = base.copy()
            for (i, j) in aset:
                m[i, j] += self.attack_delta
            return m

        def total_time(aset):
            return float(np.sum(build_matrix(aset)))

        current_score = total_time(attacked_set)
        non_attacked  = [a for a in all_arcs if a not in attacked_set]

        improved  = True
        max_iter  = 20
        iteration = 0

        while improved and iteration < max_iter:
            improved  = False
            iteration += 1
            for arc_in in list(attacked_set):
                for arc_out in non_attacked:
                    candidate = (attacked_set - {arc_in}) | {arc_out}
                    score     = total_time(candidate)
                    if score > current_score + 1e-6:
                        attacked_set  = candidate
                        non_attacked  = [a for a in all_arcs if a not in attacked_set]
                        current_score = score
                        improved      = True
                        break
                if improved:
                    break

        return build_matrix(attacked_set)