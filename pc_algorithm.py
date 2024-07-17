import numpy as np
import networkx as nx

from itertools import combinations, permutations

from gsq.ci_tests import ci_test_bin, ci_test_dis
from gsq.gsq_testdata import bin_data, dis_data


def estimate_skeleton(data, independence_test, significance_level):
    N_nodes = data.shape[1]
    N_separation_neighbors = 0
    graph = nx.complete_graph(N_nodes)
    separation_sets_lst = [[set() for _ in range(N_nodes)] for _ in range(N_nodes)]
    remove_edges_lst = []
    while True:
        continue_loop = False
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                current_neighbors = neighbors[:]
                current_neighbors.remove(neighbor)
                if len(current_neighbors) >= N_separation_neighbors:
                    for possible_separation_lst in combinations(
                        current_neighbors, N_separation_neighbors
                    ):
                        p_value = independence_test(
                            data, node, neighbor, set(possible_separation_lst)
                        )
                        if p_value > significance_level:
                            remove_edges_lst.append([node, neighbor])
                            separation_sets_lst[node][neighbor] |= set(
                                possible_separation_lst
                            )
                            separation_sets_lst[neighbor][node] |= set(
                                possible_separation_lst
                            )
                            continue_loop = True
        graph.remove_edges_from(remove_edges_lst)
        N_separation_neighbors += 1
        if not continue_loop:
            break
    return graph, separation_sets_lst


def has_both_edges(dag, node_1, node_2):
    return dag.has_edge(node_1, node_2) and dag.has_edge(node_2, node_1)


def has_any_edge(dag, node_1, node_2):
    return dag.has_edge(node_1, node_2) or dag.has_edge(node_2, node_1)


def has_one_edge(dag, node_1, node_2):
    return (
        (dag.has_edge(node_1, node_2) and (not dag.has_edge(node_2, node_1)))
        or (not dag.has_edge(node_1, node_2))
        and dag.has_edge(node_2, node_1)
    )


def has_no_edge(dag, node_1, node_2):
    return (not dag.has_edge(node_1, node_2)) and (not dag.has_edge(node_2, node_1))


def estimate_cpdag(skeleton_graph, separation_sets_lst):
    dag = skeleton_graph.to_directed()
    for node_1, node_2 in combinations(skeleton_graph.nodes(), 2):
        node_1_successors = set(dag.successors(node_1))
        if node_2 in node_1_successors:
            continue
        node_2_successors = set(dag.successors(node_2))
        if node_1 in node_2_successors:
            continue
        node_1_2_successors = node_1_successors & node_2_successors
        for node_1_2_successor in node_1_2_successors:
            if node_1_2_successor not in separation_sets_lst[node_1][node_2]:
                if dag.has_edge(node_1_2_successor, node_1):
                    dag.remove_edge(node_1_2_successor, node_1)
                if dag.has_edge(node_1_2_successor, node_2):
                    dag.remove_edge(node_1_2_successor, node_2)
    # For all the combination of nodes node_1 and node_2, apply the following
    # rules.
    old_dag = dag.copy()
    while True:
        for node_1, node_2 in permutations(skeleton_graph.nodes(), 2):
            # Rule 1: Orient node_1-node_2 into node_1->node_2 whenever there is an arrow node_1_predecessor->node_1
            # such that node_1_predecessor and node_2 are nonadjacent.
            #
            # Check if node_1-node_2.
            if has_both_edges(dag, node_1, node_2):
                # Look all the predecessors of node_1.
                for node_1_predecessor in dag.predecessors(node_1):
                    # Skip if there is an arrow node_1->node_1_predecessor.
                    if dag.has_edge(node_1, node_1_predecessor):
                        continue
                    # Skip if node_1_predecessor and node_2 are adjacent.
                    if has_any_edge(dag, node_1_predecessor, node_2):
                        continue
                    # Make node_1-node_2 into node_1->node_2
                    dag.remove_edge(node_2, node_1)
                    break
            # Rule 2: Orient node_1-node_2 into node_1->node_2 whenever there is a chain
            # node_1->node_1_predecessor->node_2.
            #
            # Check if node_1-node_2.
            if has_both_edges(dag, node_1, node_2):
                # Find nodes node_1_predecessor where node_1_predecessor is node_1->node_1_predecessor.
                succs_i = set()
                for node_1_predecessor in dag.successors(node_1):
                    if not dag.has_edge(node_1_predecessor, node_1):
                        succs_i.add(node_1_predecessor)
                # Find nodes node_2 where node_2 is node_1_predecessor->node_2.
                preds_j = set()
                for node_1_predecessor in dag.predecessors(node_2):
                    if not dag.has_edge(node_2, node_1_predecessor):
                        preds_j.add(node_1_predecessor)
                # Check if there is any node node_1_predecessor where node_1->node_1_predecessor->node_2.
                if len(succs_i & preds_j) > 0:
                    # Make node_1-node_2 into node_1->node_2
                    dag.remove_edge(node_2, node_1)
            # Rule 3: Orient node_1-node_2 into node_1->node_2 whenever there are two chains
            # node_1-node_1_predecessor->node_2 and node_1-l->node_2 such that node_1_predecessor and l are nonadjacent.
            #
            # Check if node_1-node_2.
            if has_both_edges(dag, node_1, node_2):
                # Find nodes node_1_predecessor where node_1-node_1_predecessor.
                adj_i = set()
                for node_1_predecessor in dag.successors(node_1):
                    if dag.has_edge(node_1_predecessor, node_1):
                        adj_i.add(node_1_predecessor)
                # For all the pairs of nodes in adj_i,
                for node_1_predecessor, l in combinations(adj_i, 2):
                    # Skip if node_1_predecessor and l are adjacent.
                    if has_any_edge(dag, node_1_predecessor, l):
                        continue
                    # Skip if not node_1_predecessor->node_2.
                    if dag.has_edge(node_2, node_1_predecessor) or (
                        not dag.has_edge(node_1_predecessor, node_2)
                    ):
                        continue
                    # Skip if not l->node_2.
                    if dag.has_edge(node_2, l) or (not dag.has_edge(l, node_2)):
                        continue
                    # Make node_1-node_2 into node_1->node_2.
                    dag.remove_edge(node_2, node_1)
                    break
            # Rule 4: Orient node_1-node_2 into node_1->node_2 whenever there are two chains
            # node_1-node_1_predecessor->l and node_1_predecessor->l->node_2 such that node_1_predecessor and node_2 are nonadjacent.
            #
            # However, this rule is not necessary when the PC-algorithm
            # is used to estimate a DAG.
        if nx.is_isomorphic(dag, old_dag):
            break
        old_dag = dag.copy()
    return dag


data = np.array(bin_data).reshape((5000, 5))

graph, separation_sets_lst = estimate_skeleton(
    data=data, independence_test=ci_test_bin, significance_level=0.01
)
graph = estimate_cpdag(skeleton_graph=graph, separation_sets_lst=separation_sets_lst)

graph_test = nx.DiGraph()
graph_test.add_nodes_from([0, 1, 2, 3, 4])
graph_test.add_edges_from([(0, 1), (2, 3), (3, 2), (3, 1), (2, 4), (4, 2), (4, 1)])

assert(nx.is_isomorphic(graph, graph_test))
