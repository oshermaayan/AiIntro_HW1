from framework.graph_search import *
from framework.ways import *
from .map_problem import MapProblem
from .deliveries_problem_input import DeliveriesProblemInput
from .relaxed_deliveries_problem import RelaxedDeliveriesState, RelaxedDeliveriesProblem

from typing import Set, FrozenSet, Optional, Iterator, Tuple, Union


class StrictDeliveriesState(RelaxedDeliveriesState):
    """
    An instance of this class represents a state of the strict
     deliveries problem.
    This state is basically similar to the state of the relaxed
     problem. Hence, this class inherits from `RelaxedDeliveriesState`.

    TODO:
        If you believe you need to modify the state for the strict
         problem in some sense, please go ahead and do so.
    """
    pass


class StrictDeliveriesProblem(RelaxedDeliveriesProblem):
    """
    An instance of this class represents a strict deliveries problem.
    """

    name = 'StrictDeliveries'

    def __init__(self, problem_input: DeliveriesProblemInput, roads: Roads,
                 inner_problem_solver: GraphProblemSolver, use_cache: bool = True):
        super(StrictDeliveriesProblem, self).__init__(problem_input)
        self.initial_state = StrictDeliveriesState(
            problem_input.start_point, frozenset(), problem_input.gas_tank_init_fuel)
        self.inner_problem_solver = inner_problem_solver
        self.roads = roads
        self.use_cache = use_cache
        self._init_cache()

    def _init_cache(self):
        self._cache = {}
        self.nr_cache_hits = 0
        self.nr_cache_misses = 0

    def _insert_to_cache(self, key, val):
        if self.use_cache:
            self._cache[key] = val

    def _get_from_cache(self, key):
        if not self.use_cache:
            return None
        if key in self._cache:
            self.nr_cache_hits += 1
        else:
            self.nr_cache_misses += 1
        return self._cache.get(key)

    def expand_state_with_costs(self, state_to_expand: GraphProblemState) -> Iterator[Tuple[GraphProblemState, float]]:
        """
        TODO: implement this method!
        This method represents the `Succ: S -> P(S)` function of the strict deliveries problem.
        The `Succ` function is defined by the problem operators as shown in class.
        The relaxed problem operators are defined in the assignment instructions.
        It receives a state and iterates over the successor states.
        Notice that this is an *Iterator*. Hence it should be implemented using the `yield` keyword.
        For each successor, a pair of the successor state and the operator cost is yielded.
        """
        assert isinstance(state_to_expand, StrictDeliveriesState)

        expanded_state_junction = state_to_expand.current_location

        # Explore all possible operators
        for successor_state_junction in self.possible_stop_points:
            if successor_state_junction == expanded_state_junction:
                # Don't expand your current location
                continue

            if successor_state_junction in state_to_expand.dropped_so_far:
                # By definition, operators are not defined to drop points we already visited
                continue

            # Look for cost in cache:
            cache_key = ((expanded_state_junction.index, successor_state_junction.index))
            cache_value = self._get_from_cache(cache_key)
            if cache_value is not None:
                # wanted cost is already in cache
                cost = cache_value

            else:
                # Calculate cost using inner_solver on MapProblem
                inner_problem = MapProblem(self.roads,
                                       expanded_state_junction.index,
                                       successor_state_junction.index)
                inner_problem_sol = self.inner_problem_solver.solve_problem(inner_problem)
                cost = inner_problem_sol.final_search_node.cost

                self._insert_to_cache(cache_key, cost)

            if state_to_expand.fuel < cost:
                continue

            # Check the kind of state we extend
            if successor_state_junction in self.drop_points:
                successor_state_fuel = state_to_expand.fuel - cost
                successor_state_dropped_points = state_to_expand.dropped_so_far.union(
                    set([successor_state_junction]))

            else:
                # Junction is a gas station
                successor_state_fuel = self.gas_tank_capacity
                successor_state_dropped_points = state_to_expand.dropped_so_far

            successor_state = StrictDeliveriesState(successor_state_junction,
                                                        successor_state_dropped_points,
                                                        successor_state_fuel)
            yield successor_state, cost

    def is_goal(self, state: GraphProblemState) -> bool:
        """
        This method receives a state and returns whether this state is a goal.
        A state goal is where
        """
        assert isinstance(state, StrictDeliveriesState)
        return state.dropped_so_far == self.drop_points
