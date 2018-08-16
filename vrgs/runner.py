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

import vrgs.globals as globals
import vrgs.partitions as partitions
import vrgs.full_info as full_info
import vrgs.part_info as part_info
import vrgs.no_info as no_info
import vrgs.analysis as analysis
import vrgs.MDL as MDL


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


def deduplicate_rules(vrg_rules):
    """
    Deduplicates the list of VRG rules. Two rules are 'equal' if they have the same LHS
    :param vrg_rules: list of Rule objects
    :return: rule_dict: list of unique Rules with updated frequencies
    """
    rule_dict = {}
    iso_count = 0

    for rule in vrg_rules:
        rule = deepcopy(rule)
        if rule.lhs not in rule_dict:   # first occurence of a LHS
            rule_dict[rule.lhs] = []

        isomorphic = False
        for existing_rule in rule_dict[rule.lhs]:
            if existing_rule == rule:  # isomorphic
                existing_rule.frequency += 1
                isomorphic = True
                iso_count += 1
                break   # since the new rule can only be isomorphic to exactly 1 existing rule
        if not isomorphic:
            rule_dict[rule.lhs].append(rule)

    if iso_count > 0:
        print('{} isomorphic rules'.format(iso_count))

    dedup_rules = []

    [dedup_rules.extend(v) for v in rule_dict.values()]

    return rule_dict, dedup_rules


def extract_and_generate(g, k, tree, mode='FULL', num_graphs=5):
    """
    Runs a single run of extraction and generation
    :param g: the original graph
    :param k: parameter for cond tree
    :param mode: FULL, PART, or NO
    :return: h, vrg_list
    """
    tree_time = time()
    globals.original_graph = deepcopy(g)
    if mode == 'FULL':
        print('\nUsing FULL boundary information!\n')
        extract_vrg = full_info.extract_vrg
        generate_graph = full_info.generate_graph

    elif mode == 'PART':
        print('\nUsing PARTIAL boundary information!\n')
        extract_vrg = part_info.extract_vrg
        generate_graph = part_info.generate_graph

    else:
        print('\nUsing NO boundary information!\n')
        extract_vrg = no_info.extract_vrg
        generate_graph = no_info.generate_graph

    print('k =', k)
    print('Original graph: n = {}, m = {}'.format(g.order(), g.size()))

    vrg_time = time()
    vrg_rules = extract_vrg(g, tree=[tree], lvl=0)

    print('VRG extracted in {} sec'.format(time() - vrg_time))
    print('#VRG rules: {}'.format(len(vrg_rules)))

    graphs = []
    for i in range(num_graphs):
        gen_time = time()
        h = generate_graph(vrg_rules)
        print('Generated graph #{}: n = {}, m = {}, time = {} sec'.format(i+1, h.order(), h.size(), time() - gen_time))
        graphs.append(h)

    print('total time: {} sec'.format(time() - tree_time))
    return graphs


def main():
    """
    Driver method for VRG
    :return:
    """
    # g = get_graph()
    # print(analysis.hop_plot(g))
    # return

    np.seterr(all='ignore')
    names = ['karate', 'lesmis', ]#'football', 'eucore', 'GrQc', 'bitcoin_alpha']

    with open('./stats.csv', 'w') as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['', '', '',
                            'actual graph', '',
                            'full info', '',
                            'part info', '',
                            'no info', '',
                            '', '', '', '',
                            '', '', '',
                            '', '', '',
                            '', '', ''])

        csvwriter.writerow(['graph name', 'count', 'k',
                            'n', 'm',
                            'n', 'm',
                            'n', 'm',
                            'n', 'm',
                            'mdl_graph', 'mdl_full', 'mdl_part', 'mdl_no',
                            'gcd_full', 'CVM_full_degree', 'CVM_full_PR',
                            'gcd_part', 'CVM_part_degree', 'CVM_part_PR',
                            'gcd_no', 'CVM_no_degree', 'CVM_no_PR'])

    for name in names:
        # g = get_graph()
        g = get_graph('./tmp/{}.g'.format(name))
        g.name = name
        k = 4
        globals.original_graph = deepcopy(g)


        graphs = {}
        mdl_scores = {}
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

            mdl = 0

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
        print('processing', globals.original_graph.name)

        for g_full, g_part, g_no, \
           g_mdl, mdl_full, mdl_part, mdl_no in zip(graphs['FULL'], graphs['PART'], graphs['NO'],
                                                graph_mdl, mdl_scores['FULL'], mdl_scores['PART'], mdl_scores['NO']):
            analysis.compare_graphs(g_true=globals.original_graph, g_full=g_full, g_part=g_part, g_no=g_no,
                                    graph_mdl=g_mdl, mdl_full=mdl_full, mdl_part=mdl_part, mdl_no=mdl_no,
                                    k=k, count=count)
            print(count)
            count += 1

    return


if __name__ == '__main__':
    main()