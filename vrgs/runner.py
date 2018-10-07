"""
Runner script for VRGs
1. Reads graph
2. Partitions it using either (a) Conductance, or (b) node2vec
3. Extracts grammar using one of three approaches (a) full info, (b) partial info, and (c) no info
4. Analyzes the grammar and prune rules.
4. Generates the graph with generator corresponding to (a), (b), or (c).
5. Analyzes the final network
"""

from time import time
import networkx as nx
import csv
from copy import deepcopy
import os
import sys
import numpy as np

sys.path.extend([os.getcwd(), os.path.dirname(os.getcwd())])

import vrgs.partitions as partitions
from vrgs.funky_extract import funky_extract
from vrgs.Tree import create_tree


def get_graph(filename='sample'):
    if filename == 'sample':
        g = nx.MultiGraph()
        g.add_edges_from([(1, 2), (1, 3), (1, 5),
                          (2, 4), (2, 5), (2, 7),
                          (3, 4), (3, 5),
                          (4, 5), (4, 9),
                          (6, 7), (6, 8), (6, 9),
                          (7, 8), (7, 9),
                          (8, 9)])
    elif filename == 'BA':
        g = nx.barabasi_albert_graph(10, 2, seed=42)
        g = nx.MultiGraph(g)
    else:
        g = nx.read_edgelist(filename, nodetype=int, create_using=nx.MultiGraph())
        if not nx.is_connected(g):
            g = max(nx.connected_component_subgraphs(g), key=len)
        g = nx.convert_node_labels_to_integers(g)
    return g


def get_rule_graph(rule_dict):
    g = nx.DiGraph()

    stack = [0]
    visited = set()

    while len(stack) != 0:
        u = stack.pop()
        for rule in rule_dict[u]:
            for node, d in rule.graph.nodes_iter(data=True):
                if 'label' in d:
                    v = d['label']

                    if v not in visited:
                        visited.add(v)
                        g.add_edge(u, v)
                        stack.append(v)
    print(g.edges())
    print('cycles', len(list(nx.simple_cycles(g))))
    return g


def main():
    """
    Driver method for VRG
    :return:
    """
    np.seterr(all='ignore')


    # networkx to matrix
    # np.savetxt('karate.mat', nx.to_numpy_matrix(g), fmt='%d')
    # columns = ['name', 'k', 'count',
    #            'actual_n', 'actual_m', 'actual_MDL',
    #            'full_n', 'full_m', 'full_rules', 'full_MDL', 'GCD_full', 'CVM_full_d', 'CVM_full_pr',
    #            'part_n', 'part_m', 'part_rules', 'part_MDL', 'GCD_part', 'CVM_part_d', 'CVM_part_pr',
    #            'no_n', 'no_m', 'no_rules', 'no_MDL', 'GCD_no', 'CVM_no_d', 'CVM_no_pr']

    names = ('karate', 'lesmis', 'football', 'eucore', 'GrQc', 'gnutella', 'wikivote')

    for name in names[: 3]:
        print()
        print(name)

        g = get_graph('./tmp/{}.g'.format(name))
        # g = get_graph()
        orig_graph = deepcopy(g)

        tree = partitions.approx_min_conductance_partitioning(g)
        # tree = [[[[6], [7]], [[8], [9]]], [[[5], [[3], [4]]], [[1], [2]]]]

        root = create_tree(tree)

        for k in range(3, 7):
            print('\nk =', k)

            for mode in ('full', 'part', 'no'):
                grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, pick_randomly=True)
                print('\n"{}" original graph {} nodes {} edges'.format(mode, orig_graph.order(), orig_graph.size()))

                graphs = grammar.generate_graphs(count=5)



if __name__ == '__main__':
    main()
