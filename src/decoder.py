"""
decoder.py
----------
Greedy Decoder — fast replacement for the exact MILP in the paper.

ROLE IN THE SYSTEM:
    The decoder takes a Solution (chromosome) and computes its fitness.
    It replaces the expensive MILP solver used in the paper with a
    fast greedy algorithm, making the metaheuristic practical for
    medium and large instances.

HOW IT MAPS TO THE PAPER:
    Paper component                     Our approximation
    ─────────────────────────────────   ──────────────────────────────────
    Recourse MILP (Eqs 15-36)          Greedy passenger-to-hub assignment
    CCG attacker subproblem            Greedy: attack Gamma longest arcs
    Tin/Tout timing (Eqs 30-33)        Simplified sum of distances
    ExpectedTime = sum(ps * ttime_s)   Same formula, greedy recourse
    ExpectedPenalty = worst-case        Worst-case across scenarios

SCENARIO STRUCTURE:
    Each scenario s has:
        probability  (ps)       : weight in the expected-value objective
        n_passengers (ds,p)     : number of active passengers
        omega                   : traffic congestion multiplier

    This matches the paper's formulation where each scenario has its own
    demand ds,p, not just a traffic scaling factor.

PENALTY DESIGN:
    - ExpectedTime    uses probability-weighted sum  (Eq. 4 in paper)
    - ExpectedPenalty uses WORST-CASE across scenarios (not weighted)

    Why worst-case for penalty?
    If penalty were probability-weighted, a low-probability heavy scenario
    (e.g. prob=0.15) would contribute only 15% of its penalty to the
    objective. The algorithm could then ignore it and leave passengers
    unserved in that scenario. Worst-case forces the algorithm to buy
    enough EVs to serve all passengers even in the hardest scenario.

PENALTY CALIBRATION:
    penalty_unserved must be greater than vehicle_cost to force feasibility.
    With vehicle_cost=8000 and capacity=8:
        1 unserved passenger -> penalty = penalty_unserved
        1 extra EV           -> cost    = vehicle_cost = 8000
    So penalty_unserved > 8000 makes buying an extra EV always worthwhile.
    We use 9000 as the default.
"""

import numpy as np
from typing import Tuple, List, Dict
from .solution import Solution


class Decoder:
    """
    Evaluates a Solution and fills its fitness fields.

    Parameters
    ----------
    distance_matrix  : (H, H) float  — nominal travel time t_ij between hubs
    walk_time_full   : (P_max, H) float — walk time w_{p,h} for all passengers
    recharge_time    : (H,) float    — recharge delay r_h at hub h
    hub_cost         : (H,) float    — fixed opening cost C_h per hub
    vehicle_cost     : float         — fixed activation cost C_v per EV
    capacity         : int           — seats per EV (Q in the paper)
    penalty_unserved : float         — penalty per unserved passenger (C_{p,s})
    battery_range    : float         — max EV travel distance on a full charge
    scenarios        : list of dict  — each dict has: probability, n_passengers, omega
    attack_budget    : int           — Gamma: max arcs the adversary can attack
    attack_delta     : float         — Delta: extra delay added per attacked arc
    lambda_time      : float         — lambda: monetary value of one time unit
    """

    def __init__(
        self,
        distance_matrix:  np.ndarray,
        walk_time_full:   np.ndarray,
        recharge_time:    np.ndarray,
        hub_cost:         np.ndarray,
        vehicle_cost:     float,
        capacity:         int,
        penalty_unserved: float,
        battery_range:    float      = 25.0,
        scenarios:        List[Dict] = None,
        attack_budget:    int        = 0,
        attack_delta:     float      = 5.0,
        lambda_time:      float      = 1.0,
    ):
        self.distance_matrix  = distance_matrix.astype(float)
        self.walk_time_full   = walk_time_full.astype(float)
        self.recharge_time    = recharge_time.astype(float)
        self.hub_cost         = hub_cost.astype(float)
        self.vehicle_cost     = float(vehicle_cost)
        self.capacity         = int(capacity)
        self.penalty_unserved = float(penalty_unserved)
        self.battery_range    = float(battery_range)
        self.attack_budget    = int(attack_budget)
        self.attack_delta     = float(attack_delta)
        self.lambda_time      = float(lambda_time)
        self.num_hubs         = distance_matrix.shape[0]

        # Default: single scenario with no traffic variation
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
        Evaluate a solution and return its scalar fitness.

        Steps:
          1. Structural check: abort early if no hub or no EV
          2. Investment cost: sum(Ch * yh) + K * Cv   (Eq. 3 in paper)
          3. For each scenario s:
               a. Get walk times for the n_passengers active in this scenario
               b. Assign each passenger to best open hub (greedy, Eqs 11-14)
               c. Apply adversarial attack to distance matrix
               d. Compute recourse time and penalty (greedy, Eqs 15-36)
          4. ExpectedTime    = sum_s(ps * rec_time_s)     (Eq. 4 in paper)
             ExpectedPenalty = max_s(rec_penalty_s)       (worst-case, see module doc)
          5. fitness = InvestCost + lambda*ExpectedTime + ExpectedPenalty  (Eq. 7)
        """

        # Step 1: fast structural check
        if not solution.is_structurally_feasible():
            solution.is_feasible = False
            solution.fitness     = float('inf')
            return float('inf')

        opened = solution.opened_hubs

        # Step 2: first-stage investment cost (same in all scenarios)
        solution.invest_cost = (
            float(np.dot(solution.Y, self.hub_cost))
            + solution.K * self.vehicle_cost
        )

        # Step 3 + 4: evaluate all scenarios
        expected_time = 0.0
        worst_penalty = 0.0    # worst-case (not probability-weighted)
        all_feasible  = True

        for s in self.scenarios:
            ps    = s["probability"]    # scenario probability
            n_p   = s["n_passengers"]   # active passengers in this scenario
            omega = s["omega"]          # traffic congestion multiplier

            # Walk time matrix for active passengers only: shape (n_p, H)
            walk_s = self.walk_time_full[:n_p, :]

            # Greedy passenger-to-hub assignment (approximates Eqs 11-14)
            origin_hubs, dest_hubs = self._assign_passengers(opened, walk_s, n_p)

            # Adversarial attack on distance matrix (approximates CCG subproblem)
            attacked_dist = self._apply_attack(self.distance_matrix, omega)

            # Greedy recourse evaluation (approximates Eqs 15-36)
            rec_time, rec_penalty = self._evaluate_recourse(
                solution, origin_hubs, dest_hubs, attacked_dist, omega, walk_s, n_p
            )

            # Probability-weighted time accumulation (Eq. 4 in paper)
            expected_time += ps * rec_time

            # Worst-case penalty across all scenarios (not weighted)
            if rec_penalty > worst_penalty:
                worst_penalty = rec_penalty

            if rec_penalty > 0:
                all_feasible = False

        # Step 5: assemble fitness
        solution.expected_time    = expected_time
        solution.expected_penalty = worst_penalty
        solution.is_feasible      = all_feasible

        # TotalMonetary = InvestCost + lambda*ExpectedTime + ExpectedPenalty (Eq. 7)
        solution.fitness = (
            solution.invest_cost
            + self.lambda_time * solution.expected_time
            + solution.expected_penalty
        )

        return solution.fitness

    # ------------------------------------------------------------------
    # PASSENGER ASSIGNMENT  (approximates Eqs 11-14 in paper)
    # ------------------------------------------------------------------

    def _assign_passengers(
        self, opened: np.ndarray, walk_s: np.ndarray, n_p: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assign each passenger to the open hub with minimum walk time.

        The paper defines sets Horig(p) and Hdest(p) per passenger.
        We simplify: any open hub can be origin or destination.
        Origin is the closest open hub; destination is the second closest.

        Parameters
        ----------
        opened : indices of open hubs
        walk_s : (n_p, H) walk time matrix for this scenario's passengers
        n_p    : number of active passengers in this scenario

        Returns
        -------
        origin_hubs : (n_p,) best origin hub index for each passenger
        dest_hubs   : (n_p,) best destination hub index for each passenger
        """
        if len(opened) == 0:
            zeros = np.zeros(n_p, dtype=int)
            return zeros, zeros

        # Walk times to open hubs only: shape (n_p, |opened|)
        walk_open   = walk_s[:, opened]
        best_idx    = np.argmin(walk_open, axis=1)
        origin_hubs = opened[best_idx]
        dest_hubs   = opened[best_idx]

        # Pick a different hub for destination if more than one hub is open
        if len(opened) > 1:
            walk_copy = walk_open.copy()
            walk_copy[np.arange(n_p), best_idx] = np.inf
            second_idx = np.argmin(walk_copy, axis=1)
            dest_hubs  = opened[second_idx]

        return origin_hubs, dest_hubs

    # ------------------------------------------------------------------
    # ADVERSARIAL ATTACK  (approximates CCG attacker subproblem)
    # ------------------------------------------------------------------

    def _apply_attack(self, dist: np.ndarray, omega: float) -> np.ndarray:
        """
        Simulate the adversary slowing down Gamma arcs.

        Paper: attacker solves a MILP to find the worst Gamma arcs.
        Our approximation: scale all distances by omega (traffic),
        then add attack_delta to the Gamma arcs with highest travel time.

        This greedy attacker is a lower bound on the true worst case.
        The real worst-case arcs may not be the longest ones (e.g. a
        short arc that is the only path between two hub clusters could
        be more damaging). This is a known limitation of our approach.

        Returns
        -------
        attacked distance matrix of shape (H, H)
        """
        attacked = dist * omega    # apply traffic multiplier

        if self.attack_budget > 0:
            flat     = attacked.flatten().copy()
            diag_idx = np.arange(0, len(flat), self.num_hubs + 1)
            flat[diag_idx] = -np.inf    # exclude self-loops
            top_arcs  = np.argsort(flat)[-self.attack_budget:]
            flat_full = attacked.flatten()
            flat_full[top_arcs] += self.attack_delta
            attacked  = flat_full.reshape(self.num_hubs, self.num_hubs)

        return attacked

    # ------------------------------------------------------------------
    # RECOURSE EVALUATION  (approximates Eqs 15-36 in paper)
    # ------------------------------------------------------------------

    def _evaluate_recourse(
        self,
        solution:    Solution,
        origin_hubs: np.ndarray,
        dest_hubs:   np.ndarray,
        dist:        np.ndarray,
        omega:       float,
        walk_s:      np.ndarray,
        n_p:         int,
    ) -> Tuple[float, float]:
        """
        Compute recourse time and penalty for one (scenario, attack) pair.

        For each passenger p:
            completion_time = walk[p, origin]
                            + drive_time(origin -> dest)
                            + walk[p, dest]

        drive_time:
            - If dist[origin, dest] <= battery_range: direct trip
            - Else: find best intermediate hub for recharge (Eq. 31)
            - If no recharge route exists: direct dist + 500 (death penalty)

        Fleet capacity (Eq. 17 aggregated):
            max_served  = K * capacity
            n_unserved  = max(0, n_p - max_served)
            rec_penalty = n_unserved * penalty_unserved

        Returns
        -------
        rec_time    : total passenger completion time for this scenario
        rec_penalty : unserved passenger penalty for this scenario
        """
        opened   = solution.opened_hubs
        rec_time = 0.0

        for p in range(n_p):
            o = origin_hubs[p]
            d = dest_hubs[p]

            # Walk time at boarding and alighting (w_{p,o} + w_{p,d} in paper)
            p_walk = float(walk_s[p, o] + walk_s[p, d]) * omega

            direct = dist[o, d]

            if direct <= self.battery_range:
                # EV can reach destination directly without recharging
                drive_time = direct
            else:
                # Need intermediate recharge hub (delta_{v,m}=1 in paper)
                drive_time = self._find_recharge_route(o, d, dist, opened, omega)

            rec_time += p_walk + drive_time

        # Capacity check
        max_served  = solution.K * self.capacity
        n_unserved  = max(0, n_p - max_served)
        rec_penalty = n_unserved * self.penalty_unserved

        return rec_time, rec_penalty

    # ------------------------------------------------------------------
    # RECHARGE ROUTE FINDER
    # ------------------------------------------------------------------

    def _find_recharge_route(
        self, o: int, d: int, dist: np.ndarray, opened: np.ndarray, omega: float
    ) -> float:
        """
        Find the best intermediate hub m for an EV that cannot travel o->d directly.

        Constraints (both legs must be within battery range):
            dist[o, m] <= battery_range
            dist[m, d] <= battery_range

        Total time = dist[o, m] + recharge_time[m] * omega + dist[m, d]
        This corresponds to Eq. 31 (recharge constraint) in the paper.

        Returns
        -------
        Best total drive time, or dist[o, d] + 500 if no valid route exists.
        """
        best = float('inf')

        for m in opened:
            if m == o or m == d:
                continue
            leg1 = dist[o, m]
            leg2 = dist[m, d]
            if leg1 <= self.battery_range and leg2 <= self.battery_range:
                total = leg1 + float(self.recharge_time[m]) * omega + leg2
                if total < best:
                    best = total

        # If no feasible intermediate hub found, apply heavy penalty
        return best if best < float('inf') else dist[o, d] + 500.0