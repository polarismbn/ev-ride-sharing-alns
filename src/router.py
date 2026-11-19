"""
router.py
---------
Greedy per-EV route builder with Tin/Tout timing + proper battery SoC tracking.

CHANGES IN THIS VERSION (Battery / SoC):
    - Each EV starts a route with a FULL battery (SoC = battery_range).
    - SoC is decreased by the length of every travelled leg.
    - A recharge stop is inserted ONLY when the remaining SoC is insufficient
      for the next intended leg.
    - After recharging at an intermediate hub, SoC is restored to full.
    - This replaces the previous (incorrect) check that only looked at
      whether a single leg exceeded battery_range in isolation.

Timing (still approximates Eqs 30-33 of the paper):
    Tin[v,j]  = Tout[v,i] + t_ij
    Tout[v,h] = max(Tin[v,h] + r_h, earliest boarding time)
    Ap        = Tin[v, dest] + walk[p, dest] * omega
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class EVRoute:
    """
    Complete route and timing for one EV.

    Attributes
    ----------
    ev_id       : EV index
    stops       : ordered list of hub indices visited
    boarded     : hub -> list of passengers that boarded here
    alighted    : hub -> list of passengers that alighted here
    Tin         : hub -> arrival time
    Tout        : hub -> departure time
    completion  : passenger -> completion time Ap
    total_time  : last departure time in the route
    final_soc   : remaining battery at the end of the route (for debugging)
    """
    ev_id:      int
    stops:      List[int]              = field(default_factory=list)
    boarded:    Dict[int, List[int]]   = field(default_factory=dict)
    alighted:   Dict[int, List[int]]   = field(default_factory=dict)
    Tin:        Dict[int, float]       = field(default_factory=dict)
    Tout:       Dict[int, float]       = field(default_factory=dict)
    completion: Dict[int, float]       = field(default_factory=dict)
    total_time: float                  = 0.0
    final_soc:  float                  = 0.0


class Router:
    """
    Builds greedy per-EV routes with proper Tin/Tout timing and SoC tracking.

    Parameters
    ----------
    distance_matrix      : (H, H) attacked travel time / distance matrix
    walk_time            : (n_p, H) walk times for active passengers
    recharge_time        : (H,) recharge delay per hub
    battery_range        : max EV travel distance on a full charge
    capacity             : seats per EV (Q)
    passenger_orig_hubs  : Horig(p) per passenger (optional)
    passenger_dest_hubs  : Hdest(p) per passenger (optional)
    omega                : traffic multiplier
    """

    def __init__(
        self,
        distance_matrix:     np.ndarray,
        walk_time:           np.ndarray,
        recharge_time:       np.ndarray,
        battery_range:       float,
        capacity:            int,
        passenger_orig_hubs: Optional[List] = None,
        passenger_dest_hubs: Optional[List] = None,
        omega:               float = 1.0,
    ):
        self.dist          = distance_matrix
        self.walk          = walk_time
        self.recharge      = recharge_time
        self.battery_range = float(battery_range)
        self.capacity      = int(capacity)
        self.orig_hubs     = passenger_orig_hubs
        self.dest_hubs     = passenger_dest_hubs
        self.omega         = float(omega)
        self.num_hubs      = distance_matrix.shape[0]
        self.n_passengers  = walk_time.shape[0]

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------

    def build_routes(
        self,
        opened:      np.ndarray,
        K:           int,
        origin_hubs: np.ndarray,
        dest_hubs:   np.ndarray,
    ) -> Tuple[List[EVRoute], int, float]:
        """
        Build greedy routes for K EVs.

        Returns
        -------
        routes      : list of EVRoute objects
        n_unserved  : number of passengers that could not be served
        total_Ap    : sum of all passenger completion times
        """
        assignable = set(
            p for p in range(self.n_passengers)
            if origin_hubs[p] != -1 and dest_hubs[p] != -1
        )
        unserved = set(assignable)
        routes   = []

        for ev_id in range(K):
            if not unserved:
                break

            route = self._build_single_route(
                ev_id, opened, unserved, origin_hubs, dest_hubs
            )

            if route is not None and len(route.stops) > 0:
                routes.append(route)
                for hub in route.stops:
                    for p in route.boarded.get(hub, []):
                        unserved.discard(p)

        n_unserved = len(unserved)
        total_Ap   = sum(
            ap for route in routes
            for ap in route.completion.values()
        )

        return routes, n_unserved, total_Ap

    # ------------------------------------------------------------------
    # SINGLE EV ROUTE BUILDER  (with SoC tracking)
    # ------------------------------------------------------------------

    def _build_single_route(
        self,
        ev_id:       int,
        opened:      np.ndarray,
        unserved:    set,
        origin_hubs: np.ndarray,
        dest_hubs:   np.ndarray,
    ) -> Optional[EVRoute]:
        """
        Build a greedy route for one EV.

        Battery / SoC rules:
            - EV starts with a full battery (SoC = battery_range).
            - Every leg decreases SoC by the travel distance.
            - A recharge stop is inserted only when current SoC is
              insufficient for the next intended leg.
            - After recharging, SoC is restored to full.
        """
        route    = EVRoute(ev_id=ev_id)
        on_board = []

        # Choose starting hub: open hub with most waiting passengers
        hub_demand = {}
        for p in unserved:
            o = origin_hubs[p]
            if o in opened:
                hub_demand[o] = hub_demand.get(o, 0) + 1

        if not hub_demand:
            return None

        current_hub  = max(hub_demand, key=hub_demand.get)
        current_time = 0.0
        current_soc  = self.battery_range          # <<< full battery at start

        # ── First stop ───────────────────────────────────────────────
        route.stops.append(current_hub)
        route.Tin[current_hub] = 0.0

        boarded_here = self._load_passengers(
            current_hub, on_board, unserved, origin_hubs
        )
        route.boarded[current_hub] = boarded_here
        on_board += boarded_here

        # Departure time at first hub (recharge + boarding)
        min_depart = (
            float(self.recharge[current_hub]) * self.omega
            + route.Tin[current_hub]
        )
        if boarded_here:
            boarding_ready = max(
                float(self.walk[p, current_hub]) * self.omega
                for p in boarded_here
            )
            min_depart = max(min_depart, boarding_ready)

        route.Tout[current_hub] = min_depart
        current_time = min_depart

        max_stops  = self.num_hubs * 2          # safety limit
        stop_count = 0

        # ── Main routing loop ────────────────────────────────────────
        while on_board and stop_count < max_stops:
            stop_count += 1

            next_hub = self._choose_next_hub(
                current_hub, on_board, dest_hubs, opened, route.stops
            )
            if next_hub is None:
                break

            travel = float(self.dist[current_hub, next_hub])

            # ----------------------------------------------------------
            # Battery check: do we have enough SoC for this leg?
            # ----------------------------------------------------------
            if current_soc < travel - 1e-6:
                # Need to recharge before (or on the way to) next_hub
                mid, _ = self._find_recharge_stop(
                    current_hub, next_hub, opened, route.stops,
                    current_soc=current_soc
                )

                if mid is None:
                    # Cannot reach any suitable recharge hub → abort this EV
                    break

                # Travel to the intermediate recharge hub
                leg_to_mid = float(self.dist[current_hub, mid])
                if current_soc < leg_to_mid - 1e-6:
                    # Even the recharge hub is unreachable with current SoC
                    break

                t_mid_in  = current_time + leg_to_mid
                t_mid_out = t_mid_in + float(self.recharge[mid]) * self.omega

                route.stops.append(mid)
                route.Tin[mid]      = t_mid_in
                route.Tout[mid]     = t_mid_out
                route.boarded[mid]  = []
                route.alighted[mid] = []

                # Consume battery to reach mid, then fully recharge
                current_soc  = self.battery_range
                current_time = t_mid_out
                current_hub  = mid
                travel       = float(self.dist[mid, next_hub])

                # After full recharge, check again
                if current_soc < travel - 1e-6:
                    # Still impossible (should be rare with correct _find)
                    break

            # ----------------------------------------------------------
            # Now we can travel to next_hub
            # ----------------------------------------------------------
            t_arrive = current_time + travel
            current_soc -= travel                 # consume battery

            # Base departure: arrival + recharge time at this hub
            t_depart = t_arrive + float(self.recharge[next_hub]) * self.omega

            # Unload passengers whose destination is this hub
            alighted_here = [p for p in on_board if dest_hubs[p] == next_hub]
            on_board      = [p for p in on_board if p not in alighted_here]

            for p in alighted_here:
                walk_p = float(self.walk[p, next_hub]) * self.omega
                route.completion[p] = t_arrive + walk_p

            # Load more passengers if capacity remains
            if len(on_board) < self.capacity:
                new_boarded = self._load_passengers(
                    next_hub, on_board, unserved, origin_hubs
                )
            else:
                new_boarded = []

            # Wait for boarding passengers (Eq. 32 style)
            if new_boarded:
                boarding_ready = max(
                    float(self.walk[p, next_hub]) * self.omega
                    for p in new_boarded
                )
                t_depart = max(t_depart, boarding_ready)

            # Record the stop
            route.stops.append(next_hub)
            route.Tin[next_hub]      = t_arrive
            route.Tout[next_hub]     = t_depart
            route.alighted[next_hub] = alighted_here
            route.boarded[next_hub]  = new_boarded
            on_board += new_boarded

            current_time = t_depart
            current_hub  = next_hub

        # Passengers still on board at the end of the route
        for p in on_board:
            if p not in route.completion:
                d      = dest_hubs[p]
                walk_p = float(self.walk[p, d]) * self.omega
                # Approximate remaining travel
                extra  = float(self.dist[current_hub, d]) if np.isfinite(self.dist[current_hub, d]) else 0.0
                route.completion[p] = current_time + extra + walk_p

        route.total_time = current_time
        route.final_soc  = current_soc
        return route

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _load_passengers(
        self,
        hub:         int,
        on_board:    List[int],
        unserved:    set,
        origin_hubs: np.ndarray,
    ) -> List[int]:
        """Load unserved passengers waiting at this hub up to remaining capacity."""
        waiting = sorted([
            p for p in unserved
            if origin_hubs[p] == hub and p not in on_board
        ])
        available = self.capacity - len(on_board)
        return waiting[:available]

    def _choose_next_hub(
        self,
        current_hub: int,
        on_board:    List[int],
        dest_hubs:   np.ndarray,
        opened:      np.ndarray,
        visited:     List[int],
    ) -> Optional[int]:
        """
        Choose next hub: prefer the hub where most on-board passengers alight.
        Tie-break by shorter distance.
        """
        dest_count = {}
        for p in on_board:
            d = dest_hubs[p]
            if d in opened and d != current_hub:
                dest_count[d] = dest_count.get(d, 0) + 1

        candidates = [
            h for h in opened
            if h != current_hub and np.isfinite(self.dist[current_hub, h])
        ]
        if not candidates:
            return None

        max_dist = max(self.dist[current_hub, :]) + 1e-9

        def score(h):
            return dest_count.get(h, 0) - 0.1 * self.dist[current_hub, h] / max_dist

        return max(candidates, key=score)

    def _find_recharge_stop(
        self,
        o:           int,
        d:           int,
        opened:      np.ndarray,
        visited:     List[int],
        current_soc: float = None,
    ) -> Tuple[Optional[int], float]:
        """
        Find the best intermediate hub for recharging between o and d.

        A candidate mid must satisfy:
            - reachable from o with the *current* SoC
            - after a full recharge, the leg mid → d is within battery_range
        """
        if current_soc is None:
            current_soc = self.battery_range

        best_hub  = None
        best_time = float('inf')

        for m in opened:
            if m == o or m == d or m in visited:
                continue
            if not (np.isfinite(self.dist[o, m]) and np.isfinite(self.dist[m, d])):
                continue

            leg1 = float(self.dist[o, m])
            leg2 = float(self.dist[m, d])

            # Must be reachable with current remaining battery
            if leg1 > current_soc + 1e-6:
                continue
            # After full recharge, leg2 must be feasible
            if leg2 > self.battery_range + 1e-6:
                continue

            total = leg1 + float(self.recharge[m]) * self.omega + leg2
            if total < best_time:
                best_time = total
                best_hub  = m

        return best_hub, best_time