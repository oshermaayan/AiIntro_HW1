from .graph_problem_interface import *
from .best_first_search import BestFirstSearch
from typing import Optional
import numpy as np


class GreedyStochastic(BestFirstSearch):
    def __init__(self, heuristic_function_type: HeuristicFunctionType,
                 T_init: float = 1.0, N: int = 5, T_scale_factor: float = 0.95):
        # GreedyStochastic is a graph search algorithm. Hence, we use close set.
        super(GreedyStochastic, self).__init__(use_close=True)
        self.heuristic_function_type = heuristic_function_type
        self.T = T_init
        self.N = N
        self.T_scale_factor = T_scale_factor
        self.solver_name = 'GreedyStochastic (h={heuristic_name})'.format(
            heuristic_name=heuristic_function_type.heuristic_name)

    def _init_solver(self, problem: GraphProblem):
        super(GreedyStochastic, self)._init_solver(problem)
        self.heuristic_function = self.heuristic_function_type(problem)

    def _open_successor_node(self, problem: GraphProblem, successor_node: SearchNode):

        if self.open.has_state(successor_node.state):
            already_found_node_with_same_state = self.open.get_node_by_state(successor_node.state)
            if already_found_node_with_same_state.expanding_priority > successor_node.expanding_priority:
                self.open.extract_node(already_found_node_with_same_state) # remove older node from OPEN
                self.open.push_node(successor_node) # Add new node with better cost top OPEN

        elif self.close.has_state(successor_node.state):
            already_found_node_with_same_state = self.close.get_node_by_state(successor_node.state)
            if already_found_node_with_same_state.expanding_priority > successor_node.expanding_priority:
                self.close.remove_node(already_found_node_with_same_state)  # remove older node from CLOSE
                self.open.push_node(successor_node)  # Add new node with better cost to OPEN

        else: #state hasn't been developed yet - add it to OPEN
            self.open.push_node(successor_node)


    def _calc_node_expanding_priority(self, search_node: SearchNode) -> float:
        """
        Remember: `GreedyStochastic` is greedy.
        """
        h_val = self.heuristic_function.estimate(search_node.state)
        return h_val


    def _extract_next_search_node_to_expand(self) -> Optional[SearchNode]:
        """
        Extracts the next node to expand from the open queue,
         using the stochastic method to choose out of the N
         best items from open.
        TODO: implement this method!
        Use `np.random.choice(...)` whenever you need to randomly choose
         an item from an array of items given a probabilities array `p`.
        You can read the documentation of `np.random.choice(...)` and
         see usage examples by searching it in Google.
        Notice: You might want to pop min(N, len(open) items from the
                `open` priority queue, and then choose an item out
                of these popped items. The other items have to be
                pushed again into that queue.
        """
        if self.open.is_empty():
            return None

        num_nodes_to_expand = min(self.N, len(self.open))
        nodes_to_expand = []
        nodes_vals = []
        for i in range(num_nodes_to_expand):
            node = self.open.pop_next_node()
            node_val = self._calc_node_expanding_priority(node)
            if node_val < 0.001:
                #  A node with a near-zero heuristic value was found - return it
                return node
            ### Check our conditions in this function, and understand what to chang if node_val == 0
            nodes_to_expand.append(node)
            nodes_vals.append(node_val)

        nodes_vals = np.array(nodes_vals)
        ### Remove this line:
        ### nodes_vals = np.array(list(map(lambda x: self._calc_node_expanding_priority(x), nodes_to_expand)))

        min_val = np.amin(np.array(nodes_vals))
        ### Should we even consider this case, or just return the node ?

        # Probabilities array
        prob_arr = np.zeros(len(nodes_to_expand))

        exp = -1 / self.T
        squared = np.array(list(map(lambda x: (x / min_val) ** exp, nodes_vals)))
        squared_sum = np.sum(squared)

        for i in range(len(nodes_to_expand)):
            top = (float(nodes_vals[i]) / float(min_val)) ** exp
            prob_arr[i] = (float(top) / squared_sum)

        chosen_node = np.random.choice(nodes_to_expand,size=1,p=prob_arr)[0]

        #Push other nodes back to OPEN
        returned_nodes_num = 0
        for node in nodes_to_expand:
            if node!=chosen_node:
                self.open.push_node(node)
                returned_nodes_num += 1

        assert(returned_nodes_num == num_nodes_to_expand-1) ### Remove this line after debugging
        # Update T
        self.T = self.T * self.T_scale_factor
        return chosen_node