"""
genetic_algorithm.py
--------------------
Metaheuristic optimizer for the EV Hub Location Problem.

TWO OPERATING MODES:
    1. Single-objective GA
       Minimizes scalar fitness = InvestCost + lambda*ExpectedTime + WorstPenalty
       Fast and simple. Returns one best solution.

    2. NSGA-II (Pareto mode)
       Finds the Pareto frontier between InvestCost and Recourse cost.
       Returns a set of non-dominated solutions representing trade-offs.

CHROMOSOME OPERATORS:
    Crossover : Simulated Binary Crossover (SBX) — works well on real-valued genes
    Mutation  : Gaussian noise + adaptive push away from threshold 0.5
    Selection : Tournament selection (single-obj) or NSGA-II rank+crowding (Pareto)
    Elitism   : Top n_elite solutions are always carried to the next generation

POPULATION INITIALIZATION (3 strategies for diversity):
    1. Feasibility-biased : many hubs open + enough EVs to serve all passengers
    2. Fully random       : explore the complete search space
    3. Sparse             : few hubs open, explore low-cost region

RUNTIME COMPLEXITY:
    pop_size x n_generations x O(|P| x |H|^2 x |S|)
    Example: 60 pop, 50 gen, 65 passengers, 8 hubs, 3 scenarios
    = 60 x 50 x 65 x 64 x 3 ~ 37 million simple operations -> a few seconds
"""

import numpy as np
from typing import List, Tuple, Optional
from .solution import Solution
from .decoder import Decoder


class GeneticAlgorithm:
    """
    Evolves a population of Solution objects toward better fitness.

    Parameters
    ----------
    decoder        : Decoder  -- evaluates fitness of each solution
    num_hubs       : int      -- total candidate hubs |H|
    num_passengers : int      -- max passengers across all scenarios
    V_max          : int      -- maximum EVs that can be activated
    pop_size       : int      -- number of solutions per generation
    n_generations  : int      -- number of evolutionary iterations
    crossover_prob : float    -- probability of SBX crossover per pair
    mutation_prob  : float    -- probability of mutating each individual gene
    mutation_sigma : float    -- standard deviation of Gaussian mutation noise
    elite_ratio    : float    -- fraction of top solutions preserved each generation
    pareto_mode    : bool     -- True = NSGA-II, False = single-objective GA
    seed           : int      -- random seed for reproducibility
    """

    def __init__(
        self,
        decoder:        Decoder,
        num_hubs:       int,
        num_passengers: int,
        V_max:          int   = 10,
        pop_size:       int   = 50,
        n_generations:  int   = 100,
        crossover_prob: float = 0.9,
        mutation_prob:  float = 0.1,
        mutation_sigma: float = 0.15,
        elite_ratio:    float = 0.1,
        pareto_mode:    bool  = False,
        seed:           int   = 42,
    ):
        self.decoder        = decoder
        self.num_hubs       = num_hubs
        self.num_passengers = num_passengers
        self.V_max          = V_max
        self.capacity       = decoder.capacity
        self.pop_size       = pop_size
        self.n_generations  = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob  = mutation_prob
        self.mutation_sigma = mutation_sigma
        self.n_elite        = max(1, int(elite_ratio * pop_size))
        self.pareto_mode    = pareto_mode
        self.rng            = np.random.default_rng(seed)

        # Chromosome length = num_hubs hub genes + 1 vehicle gene + num_hubs priority genes
        self.chrom_length = 2 * num_hubs + 1

        # Tracking across generations
        self.best_solution:   Optional[Solution] = None
        self.fitness_history: List[float]        = []
        self.pareto_front:    List[Solution]      = []

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------

    def evolve(self) -> Solution:
        """
        Run the full evolutionary loop and return the best solution found.

        In pareto_mode, also populates self.pareto_front with non-dominated solutions.
        """
        print(f"\n{'='*55}")
        print(f"  GA Starting | pop={self.pop_size} | gen={self.n_generations}")
        print(f"  Hubs={self.num_hubs} | Passengers={self.num_passengers} | V_max={self.V_max}")
        print(f"  Mode={'NSGA-II (Pareto)' if self.pareto_mode else 'Single-Objective'}")
        print(f"{'='*55}\n")

        # Initialize and evaluate starting population
        population = self._initialize_population()
        population = self._evaluate_all(population)

        for gen in range(1, self.n_generations + 1):

            # Selection
            if self.pareto_mode:
                population = self._nsga2_sort(population)
                parents    = population[:self.pop_size // 2]
            else:
                parents = self._tournament_select(population, self.pop_size // 2)

            # Reproduction: crossover + mutation
            offspring = self._reproduce(parents, target=self.pop_size - self.n_elite)
            offspring = self._evaluate_all(offspring)

            # Elitism: keep top n_elite from current generation
            elites     = sorted(population, key=lambda s: s.fitness)[:self.n_elite]
            elites     = [s.clone() for s in elites]
            population = elites + offspring

            # Track best solution across all generations
            best_gen = min(population, key=lambda s: s.fitness)
            self.fitness_history.append(best_gen.fitness)

            if self.best_solution is None or best_gen.fitness < self.best_solution.fitness:
                self.best_solution = best_gen.clone()

            # Progress report every 10 generations
            if gen % 10 == 0 or gen == 1:
                n_feasible = sum(1 for s in population if s.is_feasible)
                print(
                    f"  Gen {gen:>4}/{self.n_generations} | "
                    f"Best={best_gen.fitness:>12.1f} | "
                    f"Feasible={n_feasible}/{self.pop_size}"
                )

        # Extract Pareto front in NSGA-II mode
        if self.pareto_mode:
            self.pareto_front = self._extract_pareto_front(population)
            print(f"\n  Pareto front size: {len(self.pareto_front)} solutions")

        print(f"\n  Best solution: {self.best_solution}")
        return self.best_solution

    # ------------------------------------------------------------------
    # POPULATION INITIALIZATION
    # ------------------------------------------------------------------

    def _initialize_population(self) -> List[Solution]:
        """
        Build the starting population using three strategies:

        Strategy 1 (1/3 of pop) -- Feasibility-biased:
            Open 3 to num_hubs hubs with high gene values (0.65-1.0).
            Set vehicle gene high enough to serve all passengers.
            Goal: ensure feasible solutions appear from generation 1.

        Strategy 2 (1/3 of pop) -- Fully random:
            All genes drawn uniformly from [0, 1].
            Goal: explore the full search space without bias.

        Strategy 3 (1/3 of pop) -- Sparse:
            Open only 2-3 hubs. Low vehicle gene.
            Goal: explore low-cost, likely infeasible region of search space.
        """
        population = []
        third = self.pop_size // 3

        # Minimum vehicle gene ratio to serve the most demanding scenario
        min_k_ratio = min(1.0, self.num_passengers / (self.capacity * self.V_max))

        # Strategy 1: feasibility-biased
        for _ in range(third):
            chrom    = self.rng.random(self.chrom_length)
            n_open   = self.rng.integers(3, self.num_hubs + 1)
            open_idx = self.rng.choice(self.num_hubs, size=n_open, replace=False)
            chrom[:self.num_hubs]  = 0.1
            chrom[open_idx]        = 0.65 + 0.35 * self.rng.random(n_open)
            chrom[self.num_hubs]   = min_k_ratio + (1.0 - min_k_ratio) * self.rng.random()
            population.append(Solution.from_array(chrom, self.num_hubs, self.V_max))

        # Strategy 2: fully random
        for _ in range(third):
            population.append(Solution.random(self.num_hubs, self.V_max, self.rng))

        # Strategy 3: sparse (low cost exploration)
        for _ in range(self.pop_size - 2 * third):
            chrom    = self.rng.random(self.chrom_length)
            n_open   = self.rng.integers(2, 4)
            open_idx = self.rng.choice(self.num_hubs, size=n_open, replace=False)
            chrom[:self.num_hubs] = 0.1
            chrom[open_idx]       = 0.6 + 0.4 * self.rng.random(n_open)
            population.append(Solution.from_array(chrom, self.num_hubs, self.V_max))

        return population

    def _evaluate_all(self, population: List[Solution]) -> List[Solution]:
        """Decode and evaluate every solution in the population."""
        for s in population:
            self.decoder.decode(s)
        return population

    # ------------------------------------------------------------------
    # SELECTION
    # ------------------------------------------------------------------

    def _tournament_select(
        self, population: List[Solution], n: int, k: int = 3
    ) -> List[Solution]:
        """
        k-tournament selection.
        Pick k random candidates, return the one with lowest fitness.
        Repeat n times.
        """
        selected = []
        pop_arr  = np.array(population, dtype=object)
        for _ in range(n):
            idx      = self.rng.integers(0, len(population), size=k)
            winner   = min(pop_arr[idx], key=lambda s: s.fitness)
            selected.append(winner.clone())
        return selected

    # ------------------------------------------------------------------
    # CROSSOVER
    # ------------------------------------------------------------------

    def _sbx_crossover(
        self, p1: Solution, p2: Solution, eta: float = 2.0
    ) -> Tuple[Solution, Solution]:
        """
        Simulated Binary Crossover (SBX) for real-valued chromosomes.

        SBX mimics single-point binary crossover in the continuous domain.
        Parameter eta controls the spread of offspring around parents:
            Low eta  -> offspring spread widely  (more exploration)
            High eta -> offspring stay close to parents (more exploitation)
        eta=2.0 is a standard default.

        Reference: Deb & Agrawal (1995)
        """
        c1 = p1.chromosome.copy()
        c2 = p2.chromosome.copy()

        for i in range(self.chrom_length):
            if self.rng.random() < 0.5:
                u = self.rng.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                c1[i] = 0.5 * ((1 + beta) * p1.chromosome[i] + (1 - beta) * p2.chromosome[i])
                c2[i] = 0.5 * ((1 - beta) * p1.chromosome[i] + (1 + beta) * p2.chromosome[i])

        c1 = np.clip(c1, 0.0, 1.0)
        c2 = np.clip(c2, 0.0, 1.0)
        return (
            Solution.from_array(c1, self.num_hubs, self.V_max),
            Solution.from_array(c2, self.num_hubs, self.V_max),
        )

    # ------------------------------------------------------------------
    # MUTATION
    # ------------------------------------------------------------------

    def _mutate(self, solution: Solution) -> Solution:
        """
        Gaussian mutation with two adaptive repairs:

        1. Standard mutation:
           Add Gaussian noise N(0, sigma) to each gene with prob mutation_prob.

        2. Threshold push (hub genes only):
           If a hub gene lands in the ambiguous zone [0.4, 0.6], push it
           decisively to either <0.35 (closed) or >0.65 (open).
           This prevents solutions from hovering at the decision boundary.

        3. Feasibility repair for vehicle gene:
           If vehicle gene is too low to serve all passengers, push it up.
           Applied with probability 0.3 per mutation call.

        4. Hub diversity repair:
           If only 1 hub is open after mutation, randomly open another.
           Applied with probability 0.5.
        """
        chrom = solution.chromosome.copy()

        # Standard Gaussian mutation per gene
        for i in range(self.chrom_length):
            if self.rng.random() < self.mutation_prob:
                chrom[i] = np.clip(chrom[i] + self.rng.normal(0, self.mutation_sigma), 0.0, 1.0)

                # Push hub genes out of the ambiguous zone near threshold 0.5
                if i < self.num_hubs and 0.4 <= chrom[i] <= 0.6:
                    chrom[i] = self.rng.choice([
                        0.2 + 0.15 * self.rng.random(),   # push toward closed
                        0.65 + 0.35 * self.rng.random(),  # push toward open
                    ])

        # Repair 1: ensure vehicle gene is high enough to serve all passengers
        min_k_ratio = min(1.0, self.num_passengers / (self.capacity * self.V_max))
        if self.rng.random() < 0.3 and chrom[self.num_hubs] < min_k_ratio:
            chrom[self.num_hubs] = min_k_ratio + 0.1 * self.rng.random()

        # Repair 2: ensure at least 2 hubs are open
        hub_genes = chrom[:self.num_hubs]
        if np.sum(hub_genes > 0.5) < 2 and self.rng.random() < 0.5:
            closed = np.where(hub_genes <= 0.5)[0]
            if len(closed) > 0:
                pick = self.rng.choice(closed)
                chrom[pick] = 0.65 + 0.35 * self.rng.random()

        return Solution.from_array(chrom, self.num_hubs, self.V_max)

    # ------------------------------------------------------------------
    # REPRODUCTION
    # ------------------------------------------------------------------

    def _reproduce(self, parents: List[Solution], target: int) -> List[Solution]:
        """
        Generate offspring via crossover + mutation until target count reached.
        """
        offspring = []
        while len(offspring) < target:
            idx    = self.rng.integers(0, len(parents), size=2)
            p1, p2 = parents[idx[0]], parents[idx[1]]

            if self.rng.random() < self.crossover_prob:
                c1, c2 = self._sbx_crossover(p1, p2)
            else:
                c1, c2 = p1.clone(), p2.clone()

            offspring.append(self._mutate(c1))
            if len(offspring) < target:
                offspring.append(self._mutate(c2))

        return offspring[:target]

    # ------------------------------------------------------------------
    # NSGA-II (PARETO MODE)
    # ------------------------------------------------------------------

    def _nsga2_sort(self, population: List[Solution]) -> List[Solution]:
        """
        NSGA-II non-dominated sorting + crowding distance assignment.

        Step 1: Assign each solution a Pareto rank.
            rank 0 = non-dominated (Pareto front 1)
            rank 1 = non-dominated after removing rank 0
            etc.

        Step 2: Within each rank, assign crowding distance.
            Boundary solutions get infinite distance.
            Interior solutions get distance based on neighbor gap in objective space.

        Step 3: Sort by (rank ASC, crowding_distance DESC).
            This gives preference to non-dominated diverse solutions.

        Reference: Deb et al. (2002) -- A Fast and Elitist Multiobjective GA: NSGA-II
        """
        n = len(population)
        domination_count = np.zeros(n, dtype=int)
        dominated_by     = [[] for _ in range(n)]
        fronts           = [[]]

        # Build domination relationships
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = population[i], population[j]
                if si.dominates(sj):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif sj.dominates(si):
                    dominated_by[j].append(i)
                    domination_count[i] += 1
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(i)

        # Build subsequent fronts
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = k + 1
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        # Assign crowding distances within each front
        for front in fronts:
            if not front:
                continue
            self._assign_crowding([population[i] for i in front])

        # Sort: rank ascending, crowding distance descending
        population.sort(key=lambda s: (s.rank, -s.crowding_distance))
        return population

    def _assign_crowding(self, front: List[Solution]):
        """
        Compute crowding distance for all solutions in one Pareto front.
        Boundary solutions (min/max of each objective) get infinite distance.
        """
        n = len(front)
        if n <= 2:
            for s in front:
                s.crowding_distance = float('inf')
            return

        for s in front:
            s.crowding_distance = 0.0

        # Objective 1: invest_cost
        front.sort(key=lambda s: s.invest_cost)
        front[0].crowding_distance  = float('inf')
        front[-1].crowding_distance = float('inf')
        r1 = front[-1].invest_cost - front[0].invest_cost + 1e-9
        for i in range(1, n - 1):
            front[i].crowding_distance += (
                (front[i+1].invest_cost - front[i-1].invest_cost) / r1
            )

        # Objective 2: expected_time + expected_penalty (recourse cost)
        front.sort(key=lambda s: s.expected_time + s.expected_penalty)
        r2 = (
            (front[-1].expected_time + front[-1].expected_penalty)
            - (front[0].expected_time  + front[0].expected_penalty)
            + 1e-9
        )
        front[0].crowding_distance  = float('inf')
        front[-1].crowding_distance = float('inf')
        for i in range(1, n - 1):
            front[i].crowding_distance += (
                ((front[i+1].expected_time + front[i+1].expected_penalty)
                 - (front[i-1].expected_time + front[i-1].expected_penalty))
                / r2
            )

    def _extract_pareto_front(self, population: List[Solution]) -> List[Solution]:
        """
        Return unique non-dominated (rank=0) solutions.
        Two solutions are considered duplicates if both objectives differ by less than 1.
        """
        rank0  = [s for s in population if s.rank == 0]
        unique = []
        for s in rank0:
            is_dup = any(
                abs(s.invest_cost - u.invest_cost) < 1.0 and
                abs((s.expected_time + s.expected_penalty)
                    - (u.expected_time + u.expected_penalty)) < 1.0
                for u in unique
            )
            if not is_dup:
                unique.append(s)

        return sorted(unique, key=lambda s: s.invest_cost)