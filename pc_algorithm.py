import numpy as np
import networkx as nx

from itertools import combinations, permutations

from gsq.ci_tests import ci_test_bin, ci_test_dis
from gsq.gsq_testdata import bin_data, dis_data


def estimate_skeleton(indep_test_func, data_matrix, alpha):
    method_stable = False
    node_ids = range(data_matrix.shape[1])
    node_size = data_matrix.shape[1]
    sep_set = [[set() for i in range(node_size)] for j in range(node_size)]
    g = nx.complete_graph(node_size)
    l = 0
    while True:
        cont = False
        remove_edges = []
        for (i, j) in permutations(node_ids, 2):
            adj_i = list(g.neighbors(i))
            if j not in adj_i:
                continue
            else:
                adj_i.remove(j)
            if len(adj_i) >= l:
                for k in combinations(adj_i, l):
                    p_val = indep_test_func(data_matrix, i, j, set(k))
                    if p_val > alpha:
                        if g.has_edge(i, j):
                            if method_stable:
                                remove_edges.append((i, j))
                            else:
                                g.remove_edge(i, j)
                        sep_set[i][j] |= set(k)
                        sep_set[j][i] |= set(k)
                        break
                cont = True
        l += 1
        if method_stable:
            g.remove_edges_from(remove_edges)
        if cont is False:
            break
    return g, sep_set


# dm = np.array(bin_data).reshape((5000, 5))
# g, sep_set = estimate_skeleton(indep_test_func=ci_test_bin, data_matrix=dm, alpha=0.01)

# print(g)

def own_estimate_skeleton(data, independence_test, significance_level):
    N_nodes = data.shape[1]
    graph = nx.complete_graph(N_nodes)
    min_neighbor_nodes = 0
    for node in graph.nodes:
        print(len(list(graph.neighbors(node))))

N = 1000
M = 3
X = np.random.randint(0, 2, size=[N, M])

own_estimate_skeleton(X, ci_test_bin, 0.05)
