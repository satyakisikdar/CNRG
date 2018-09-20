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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.extend([os.getcwd(), os.path.dirname(os.getcwd())])

import vrgs.globals as globals
import vrgs.partitions as partitions
import vrgs.full_info as full_info
import vrgs.part_info as part_info
import vrgs.no_info as no_info
import vrgs.analysis as analysis
import vrgs.MDL as MDL
import vrgs.funky_extract as funky_extract

from vrgs.Tree import create_tree

from vrgs.GCD import GCD


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


def main():
    """
    Driver method for VRG
    :return:
    """
    np.seterr(all='ignore')

    # for _ in range(10):
    #     g = nx.erdos_renyi_graph(100, 0.2)
    #     h = nx.erdos_reny i_graph(100, 0.2)
    #     print(GCD(g, h, mode='rage'), GCD(g, h, mode='orca'))
    # return

    g = get_graph()
    # g = get_graph('./tmp/football.g')
    # print(nx.laplacian_spectrum(g))
    globals.original_graph = deepcopy(g)

    tree = partitions.approx_min_conductance_partitioning(g, 1)
    root = create_tree(tree)

    funky_extract.funky_runner(root, 3)

    print(root)


    return

    if len(sys.argv) < 2:
        k = 3
    else:
        k = int(sys.argv[1])

    print('k =', k)
    names = ['karate', 'lesmis', 'football', 'eucore', 'GrQc', 'gnutella', 'wikivote']

    write = False
    if write:
        with open('./stats_{}.csv'.format(k), 'w') as f:
            csvwriter = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['', '', '',
                                'actual graph', '',
                                'full info', '', '',
                                'part info', '', '',
                                'no info', '', '',
                                '', '', '', '',
                                '', '', '',
                                '', '', '',
                                '', '', ''])

            csvwriter.writerow(['graph name', 'count', 'k',
                                'n', 'm',
                                'n', 'm', 'uniq rules',
                                'n', 'm', 'uniq rules',
                                'n', 'm', 'uniq rules',
                                'mdl_graph', 'mdl_full', 'mdl_part', 'mdl_no',
                                'gcd_full', 'CVM_full_degree', 'CVM_full_PR',
                                'gcd_part', 'CVM_part_degree', 'CVM_part_PR',
                                'gcd_no', 'CVM_no_degree', 'CVM_no_PR'])


    for name in names[4: ]:

        for k in range(2, 6):
            # g = get_graph()
            g = get_graph('./tmp/{}.g'.format(name))
            g.name = name
            globals.original_graph = deepcopy(g)
            print('\nprocessing {} k = {}, n = {} m = {}'.format(globals.original_graph.name, k, globals.original_graph.order(),
                                                       globals.original_graph.size()))

            tree = partitions.approx_min_conductance_partitioning(g, k)

            graphs = {}
            mdl_scores = {}
            rule_counts = {}

            if write:
                graph_mdl = [MDL.graph_mdl_v2(globals.original_graph, l_u=2)] * 10

            for mode in ['FULL', 'PART', 'NO']:
                if mode == 'FULL':
                    extract_vrg = full_info.extract_vrg
                    generate_graph = full_info.generate_graph

                elif mode == 'PART':
                    extract_vrg = part_info.extract_vrg
                    generate_graph = part_info.generate_graph

                else:
                    extract_vrg = no_info.extract_vrg
                    generate_graph = no_info.generate_graph

                tree = partitions.approx_min_conductance_partitioning(g, k)

                vrg = extract_vrg(deepcopy(g), tree=[deepcopy(tree)], lvl=0)
                rule_dict = {}
                rule_count = 0

                for rule in vrg:
                    rule = deepcopy(rule)
                    if rule.lhs not in rule_dict:  # first occurence of a LHS
                        rule_dict[rule.lhs] = []

                    isomorphic = False
                    for existing_rule in rule_dict[rule.lhs]:
                        if existing_rule == rule:  # isomorphic
                            existing_rule.frequency += 1
                            isomorphic = True
                            break  # since the new rule can only be isomorphic to exactly 1 existing rule
                    if not isomorphic:
                        rule_dict[rule.lhs].append(rule)
                        rule_count += 1
                rule_counts[mode] = [rule_count] * 10

                print(mode, rule_count, 'rules', end=' ')
                continue
                mdl = 0

                if write:
                    print('calculating {} MDL for {}'.format(mode, g.name))

                    for rule_list in rule_dict.values():
                        for rule in rule_list:
                            rule.calculate_cost()
                            mdl += rule.cost

                    mdl_scores[mode] = [mdl] * 10

                graphs[mode] = []
                for _ in range(10):
                    h = generate_graph(rule_dict)
                    graphs[mode].append(h)

            count = 1

            if write:
                for g_full, g_part, g_no, \
                    rule_full, rule_part, rule_no, \
                   g_mdl, mdl_full, mdl_part, mdl_no \
                        in zip(graphs['FULL'], graphs['PART'], graphs['NO'],
                                rule_counts['FULL'], rule_counts['PART'], rule_counts['NO'],
                                graph_mdl, mdl_scores['FULL'], mdl_scores['PART'], mdl_scores['NO']):

                    analysis.compare_graphs(g_true=globals.original_graph, g_full=g_full, g_part=g_part, g_no=g_no,
                                            graph_mdl=g_mdl, mdl_full=mdl_full, mdl_part=mdl_part, mdl_no=mdl_no,
                                            rule_full=rule_full, rule_part=rule_part, rule_no=rule_no, k=k, count=count)
                    print(count)
                    count += 1

    return


if __name__ == '__main__':
    main()
