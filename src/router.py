"""
router.py
---------
Greedy per-EV route builder with Tin/Tout timing.

CHANGES IN THIS VERSION:
    Added proper Tin/Tout timing (approximates Eqs 30-33 in paper):

    Eq 30: Tin[v,j] >= Tout[v,i] + t_ij   (travel time constraint)
    Eq 31: Tout[v,h] >= Tin[v,h] + r_h    (recharge constraint)
    Eq 32: Tout[v,i] >= w_{p,i}            (boarding: EV waits for passenger)
    Eq 33: Ap >= Tin[v,j] + w_{p,j}        (completion time at destination)

    Previously: Ap was approximated as departure_time + travel + walk.
    Now: Ap is computed from exact Tin[v, dest_hub] + walk[p, dest_hub].

    The EV schedule is sequential:
        Tin[v, next_hub]  = Tout[v, current_hub] + travel_time
        Tout[v, next_hub] = max(Tin[v, next_hub] + recharge, earliest_boarding)
        Ap = Tin[v, dest_hub] + walk[p, dest_hub] * omega

    This gives a more accurate completion time for each passenger,
    which directly feeds into the ExpectedTime objective (Eq. 4).
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
    Tin         : hub -> arrival time (Tin_{v,h} in paper)
    Tout        : hub -> departure time (Tout_{v,h} in paper)
    completion  : passenger -> completion time Ap (Eq. 33)
    total_time  : last departure time in the route
    """
    ev_id:      int
    stops:      List[int]              = field(default_factory=list)
    boarded:    Dict[int, List[int]]   = field(default_factory=dict)
    alighted:   Dict[int, List[int]]   = field(default_factory=dict)
    Tin:        Dict[int, float]       = field(default_factory=dict)
    Tout:       Dict[int, float]       = field(default_factory=dict)
    completion: Dict[int, float]       = field(default_factory=dict)
    total_time: float                  = 0.0


class Router:
    """
    Builds greedy per-EV routes with proper Tin/Tout timing.

    Parameters
    ----------
    distance_matrix      : (H, H) attacked travel time matrix
    walk_time            : (n_p, H) walk times for active passengers
    recharge_time        : (H,) recharge delay per hub
    battery_range        : max EV travel per charge
    capacity             : seats per EV (Q)
    passenger_orig_hubs  : Horig(p) per passenger
    passenger_dest_hubs  : Hdest(p) per passenger
    omega                : traffic multiplier (applied to walk times)
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
        self.battery_range = battery_range
        self.capacity      = capacity
        self.orig_hubs     = passenger_orig_hubs
        self.dest_hubs     = passenger_dest_hubs
        self.omega         = omega
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
        n_unserved  : passengers not served
        total_Ap    : sum of all passenger completion times (Eq. 34)
        """
        # Passengers with valid assignment
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
                # Mark all boarded passengers as served
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
    # SINGLE EV ROUTE BUILDER
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
        Build a greedy route for one EV with proper Tin/Tout timing.

        Timing rules (matching paper Eqs 30-33):
            Tin[h]  = Tout[prev] + travel(prev -> h)           (Eq. 30)
            Tout[h] = max(Tin[h] + recharge[h],               (Eq. 31)
                          max(walk[p, h] for p boarding at h)) (Eq. 32)
            Ap      = Tin[dest_hub] + walk[p, dest_hub]*omega  (Eq. 33)
        """
        route    = EVRoute(ev_id=ev_id)
        on_board = []    # passengers currently in EV

        # Choose starting hub: open hub with most waiting passengers
        hub_demand = {}
        for p in unserved:
            o = origin_hubs[p]
            if o in opened:
                hub_demand[o] = hub_demand.get(o, 0) + 1

        if not hub_demand:
            return None

        current_hub  = max(hub_demand, key=hub_demand.get)
        current_time = 0.0   # EV clock starts at 0

        # ── First stop: starting hub ─────────────────────────────────
        route.stops.append(current_hub)

        # Tin = 0 at starting hub (Eq. 30, first stop)
        route.Tin[current_hub] = 0.0

        # Load passengers at starting hub
        boarded_here = self._load_passengers(
            current_hub, on_board, unserved, origin_hubs
        )
        route.boarded[current_hub] = boarded_here
        on_board += boarded_here

        # Tout[h] = max(Tin[h] + recharge, max walk of boarding passengers)
        # Eq. 31 + Eq. 32
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

        max_stops  = self.num_hubs
        stop_count = 0

        # ── Main routing loop ────────────────────────────────────────
        while on_board and stop_count < max_stops:
            stop_count += 1

            next_hub = self._choose_next_hub(
                current_hub, on_board, dest_hubs, opened, route.stops
            )

            if next_hub is None:
                break

            travel = self.dist[current_hub, next_hub]

            # If direct travel exceeds battery range, find recharge stop
            if travel > self.battery_range:
                mid, _ = self._find_recharge_stop(
                    current_hub, next_hub, opened, route.stops
                )
                if mid is None:
                    break

                # Add recharge intermediate stop
                t_mid_in  = current_time + self.dist[current_hub, mid]
                t_mid_out = t_mid_in + float(self.recharge[mid]) * self.omega

                route.stops.append(mid)
                route.Tin[mid]  = t_mid_in
                route.Tout[mid] = t_mid_out
                route.boarded[mid]  = []
                route.alighted[mid] = []

                current_time = t_mid_out
                current_hub  = mid
                travel       = self.dist[mid, next_hub]

            # ── Arrive at next hub (Eq. 30) ──────────────────────────
            t_arrive = current_time + travel

            # ── Depart time: max(recharge done, all passengers ready) ─
            # Eq. 31: must wait for recharge
            t_depart = t_arrive + float(self.recharge[next_hub]) * self.omega

            # ── Unload passengers whose destination is this hub ───────
            alighted_here = [p for p in on_board if dest_hubs[p] == next_hub]
            on_board      = [p for p in on_board if p not in alighted_here]

            # Record completion times Ap for alighted passengers (Eq. 33)
            for p in alighted_here:
                walk_p = float(self.walk[p, next_hub]) * self.omega
                Ap     = t_arrive + walk_p
                route.completion[p] = Ap

            # ── Load more passengers if capacity allows ───────────────
            if len(on_board) < self.capacity:
                new_boarded = self._load_passengers(
                    next_hub, on_board, unserved, origin_hubs
                )
            else:
                new_boarded = []

            # Eq. 32: EV must wait for boarding passengers
            if new_boarded:
                boarding_ready = max(
                    float(self.walk[p, next_hub]) * self.omega
                    for p in new_boarded
                )
                t_depart = max(t_depart, boarding_ready)

            # Record stop
            route.stops.append(next_hub)
            route.Tin[next_hub]     = t_arrive
            route.Tout[next_hub]    = t_depart
            route.alighted[next_hub] = alighted_here
            route.boarded[next_hub]  = new_boarded
            on_board += new_boarded

            current_time = t_depart
            current_hub  = next_hub

        # Any passengers still on board at end of route: record completion
        # at last hub (EV terminates route)
        for p in on_board:
            if p not in route.completion:
                d      = dest_hubs[p]
                walk_p = float(self.walk[p, d]) * self.omega
                route.completion[p] = current_time + self.dist[current_hub, d] + walk_p

        route.total_time = current_time
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
        """Load unserved passengers at this hub up to remaining capacity."""
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
        Choose next hub: prefer hub where most on-board passengers alight.
        Among ties: pick closest hub.
        """
        dest_count = {}
        for p in on_board:
            d = dest_hubs[p]
            if d in opened and d != current_hub:
                dest_count[d] = dest_count.get(d, 0) + 1

        # Allow revisiting hubs (multiple EVs can serve same hub)
        # Only exclude the current hub to prevent self-loops
        candidates = [
            h for h in opened
            if h != current_hub
        ]

        if not candidates:
            return None

        max_dist = max(self.dist[current_hub, :]) + 1e-9

        def score(h):
            return dest_count.get(h, 0) - 0.1 * self.dist[current_hub, h] / max_dist

        return max(candidates, key=score)

    def _find_recharge_stop(
        self,
        o:       int,
        d:       int,
        opened:  np.ndarray,
        visited: List[int],
    ) -> Tuple[Optional[int], float]:
        """Find best intermediate recharge hub between o and d."""
        best_hub  = None
        best_time = float('inf')

        for m in opened:
            if m == o or m == d or m in visited:
                continue
            leg1 = self.dist[o, m]
            leg2 = self.dist[m, d]
            if leg1 <= self.battery_range and leg2 <= self.battery_range:
                total = leg1 + float(self.recharge[m]) * self.omega + leg2
                if total < best_time:
                    best_time = total
                    best_hub  = m

        return best_hub, best_time