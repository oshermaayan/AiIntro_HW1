from framework import *
from deliveries import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union

# Load the map
roads = load_map_from_csv(Consts.get_data_file_path("tlv.csv"))

# Make `np.random` behave deterministic.
Consts.set_seed()


# --------------------------------------------------------------------
# -------------------------- Map Problem -----------------------------
# --------------------------------------------------------------------

def plot_distance_and_expanded_wrt_weight_figure(
        weights: Union[np.ndarray, List[float]],
        total_distance: Union[np.ndarray, List[float]],
        total_expanded: Union[np.ndarray, List[int]]):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    """
    assert len(weights) == len(total_distance) == len(total_expanded)

    fig, ax1 = plt.subplots()

    # See documentation here:
    # https://matplotlib.org/2.0.0/api/_as_gen/matplotlib.axes.Axes.plot.html
    # You can also search google for additional examples.
    plt.plot(weights, total_distance, 'b')

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('distance traveled', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()

    plt.plot(weights, total_expanded, 'r')
    #ax2.plot(weights, total_expanded)

    ax2.set_ylabel('States expanded', color='r')
    ax2.tick_params('y', colors='r')
    ax2.set_xlabel('weight')

    fig.tight_layout()
    plt.show()


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem):
    # TODO:
    # 1. Create an array of 20 numbers equally spreaded in [0.5, 1]
    #    (including the edges). You can use `np.linspace()` for that.
    # 2. For each weight in that array run the A* algorithm, with the
    #    given `heuristic_type` over the map problem. For each such run,
    #    store the cost of the solution (res.final_search_node.cost)
    #    and the number of expanded states (res.nr_expanded_states).
    #    Store these in 2 lists (array for the costs and array for
    #    the #expanded.
    # Call the function `plot_distance_and_expanded_by_weight_figure()`
    #  with that data.
    weights = np.linspace(0.5, 1, 20)
    costs_list = []
    num_expanded_list = []
    for weight in weights:
        astar = AStar(heuristic_type, heuristic_weight=weight)
        astar_res = astar.solve_problem(problem)
        costs_list.append(astar_res.final_search_node.cost)
        num_expanded_list.append(astar_res.nr_expanded_states)

    plot_distance_and_expanded_wrt_weight_figure(weights, costs_list, num_expanded_list)




def map_problem():
    print()
    print('Solve the map problem.')

    # Ex.8
    map_prob = MapProblem(roads, 54, 549)
    uc = UniformCost()
    #res = uc.solve_problem(map_prob)
    #print(res)

    # Ex.10
    # TODO: create an instance of `AStar` with the `NullHeuristic`,
    #       solve the same `map_prob` with it and print the results (as before).
    # Notice: AStar constructor receives the heuristic *type* (ex: `MyHeuristicClass`),
    #         and not an instance of the heuristic (eg: not `MyHeuristicClass()`).
    astar_solver_nullHeur = AStar(NullHeuristic, heuristic_weight=0)
    #astar_weightzero_res = astar_solver_nullHeur.solve_problem(map_prob)
   #print(astar_weightzero_res)

    # Ex.11
    # TODO: create an instance of `AStar` with the `AirDistHeuristic`,
    #       solve the same `map_prob` with it and print the results (as before).
    astar_solver_airDistHeur = AStar(AirDistHeuristic)
    #astar_AirDist_res = astar_solver_airDistHeur.solve_problem(map_prob)
    #print(astar_AirDist_res)

    # Ex.12
    # TODO:
    # 1. Complete the implementation of the function
    #    `run_astar_for_weights_in_range()` (upper in this file).
    # 2. Complete the implementation of the function
    #    `plot_distance_and_expanded_by_weight_figure()`
    #    (upper in this file).
    # 3. Call here the function `run_astar_for_weights_in_range()`
    #    with `AirDistHeuristic` and `map_prob`.
    ###run_astar_for_weights_in_range(AirDistHeuristic, map_prob)


# --------------------------------------------------------------------
# ----------------------- Deliveries Problem -------------------------
# --------------------------------------------------------------------

def relaxed_deliveries_problem():

    print()
    print('Solve the relaxed deliveries problem.')

    ### CHANGE BACK TO BIG_DELIVERY.IN
    big_delivery = DeliveriesProblemInput.load_from_file('big_delivery.in', roads)#DeliveriesProblemInput.load_from_file('big_delivery.in', roads)
    big_deliveries_prob = RelaxedDeliveriesProblem(big_delivery)

    # Ex.16
    #       solve the `big_deliveries_prob` with it and print the results (as before).
    astar_solver_maxAirDistHeur = AStar(MaxAirDistHeuristic)
    ###astar_maxAirDist_res = astar_solver_maxAirDistHeur.solve_problem(big_deliveries_prob)
    ###print(astar_maxAirDist_res)

    # Ex.17
    # TODO: create an instance of `AStar` with the `MSTAirDistHeuristic`,
    #       solve the `big_deliveries_prob` with it and print the results (as before).
    astar_relaxed_MST_solver = AStar(MSTAirDistHeuristic)
    ###astar_relaxed_MST_res = astar_relaxed_MST_solver.solve_problem(big_deliveries_prob)
    ###print(astar_relaxed_MST_res)



    # Ex.18
    # TODO: Call here the function `run_astar_for_weights_in_range()`
    #       with `MSTAirDistHeuristic` and `big_deliveries_prob`.

    ###run_astar_for_weights_in_range(MSTAirDistHeuristic, big_deliveries_prob)

    # Ex.24
    # TODO:
    # 1. Run the stochastic greedy algorithm for 100 times.
    #    For each run, store the cost of the found solution.
    #    Store these costs in a list.
    # 2. The "Anytime Greedy Stochastic Algorithm" runs the greedy
    #    greedy stochastic for N times, and after each iteration
    #    stores the best solution found so far. It means that after
    #    iteration #i, the cost of the solution found by the anytime
    #    algorithm is the MINIMUM among the costs of the solutions
    #    found in iterations {1,...,i}. Calculate the costs of the
    #    anytime algorithm wrt the #iteration and store them in a list.
    # 3. Calculate and store the cost of the solution received by
    #    the A* algorithm (with w=0.5).
    # 4. Calculate and store the cost of the solution received by
    #    the deterministic greedy algorithm (A* with w=1).
    # 5. Plot a figure with the costs (y-axis) wrt the #iteration
    #    (x-axis). Of course that the costs of A*, and deterministic
    #    greedy are not dependent with the iteration number, so
    #    these two should be represented by horizontal lines.
    ''' ### Uncomment this entire section!
    run_times = 100
    greedy_stoch_costs_arr = []

    greedy_stoch_solver = GreedyStochastic(MSTAirDistHeuristic)
    greedy_stoch_res = greedy_stoch_solver.solve_problem(big_deliveries_prob)
    min_greedy_cost = greedy_stoch_res.final_search_node.cost
    greedy_stoch_costs_arr.append(min_greedy_cost)
    best_greedy_sol = greedy_stoch_res

    greedy_stoch_anytime_costs = []
    greedy_stoch_anytime_costs.append(min_greedy_cost)

    for i in range(run_times - 1):
        greedy_stoch_res = greedy_stoch_solver.solve_problem(big_deliveries_prob)
        cost = greedy_stoch_res.final_search_node.cost
        greedy_stoch_costs_arr.append(cost)

        if cost < min_greedy_cost:
            min_greedy_cost = cost
            best_greedy_sol = greedy_stoch_res

        greedy_stoch_anytime_costs.append(min_greedy_cost)


    astar_solver = AStar(MSTAirDistHeuristic,heuristic_weight=0.5)
    greedy_deter_solver = AStar(MSTAirDistHeuristic,heuristic_weight=1)

    astar_cost = astar_solver.solve_problem(big_deliveries_prob).final_search_node.cost
    greedy_deter_cost = greedy_deter_solver.solve_problem(big_deliveries_prob).final_search_node.cost

    astar_cost_arr = [astar_cost]*run_times
    greedy_deter_cost_arr = [greedy_deter_cost] * run_times

    # Plot cost per iteration
    fig, ax1 = plt.subplots()

    x_axis = list(range(1, run_times + 1))
    plt.plot(x_axis, greedy_stoch_costs_arr, label="Greedy stochastic")
    plt.plot(x_axis, greedy_stoch_anytime_costs, label="Anytime algorithm")
    plt.plot(x_axis, astar_cost_arr, label="Astar")
    plt.plot(x_axis, greedy_deter_cost_arr, label="Greedy deterministic")

    ax1.set_ylabel('cost', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('Iteration')

    plt.title("Costs as a function of iteration")
    plt.legend()
    plt.grid()
    plt.show()
    '''




def strict_deliveries_problem():
    print()
    print('Solve the strict deliveries problem.')

    small_delivery = DeliveriesProblemInput.load_from_file('small_delivery.in', roads)
    small_deliveries_strict_problem = StrictDeliveriesProblem(
        small_delivery, roads, inner_problem_solver=AStar(AirDistHeuristic))

    # Ex.26
    # TODO: Call here the function `run_astar_for_weights_in_range()`
    #       with `MSTAirDistHeuristic` and `big_deliveries_prob`.
    run_astar_for_weights_in_range(MSTAirDistHeuristic, small_deliveries_strict_problem)

    # Ex.28
    # TODO: create an instance of `AStar` with the `RelaxedDeliveriesHeuristic`,
    #       solve the `small_deliveries_strict_problem` with it and print the results (as before).
    astar_relaxed_heur_solver = AStar(RelaxedDeliveriesHeuristic)
    astar_relaxed_heur_solution = astar_relaxed_heur_solver.solve_problem(small_deliveries_strict_problem)
    print(astar_relaxed_heur_solution)

def main():
    map_problem()
    relaxed_deliveries_problem()
    strict_deliveries_problem()


if __name__ == '__main__':
    main()
