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
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import os
import sys

sys.path.extend([os.getcwd(), os.path.dirname(os.getcwd())])

import vrgs.globals as globals
import vrgs.partitions as partitions
import vrgs.full_info as full_info
import vrgs.part_info as part_info
import vrgs.no_info as no_info
import vrgs.analysis as analysis

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
        if rule.lhs not in rule_dict:   # first occurence of a LHS
            rule_dict[rule.lhs] = []

        isomorphic = False
        for existing_rule in rule_dict[rule.lhs]:
            if existing_rule == rule:  # isomorphic
                existing_rule.frequency += 1
                isomorphic = True
                iso_count += 1
                break # since the new rule can only be isomorphic to exactly 1 existing rule
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
        print('Using PARTIAL boundary information!\n')
        extract_vrg = part_info.extract_vrg
        generate_graph = part_info.generate_graph

    else:
        print('Using NO boundary information!\n')
        extract_vrg = no_info.extract_vrg
        generate_graph = no_info.generate_graph

    print('k =', k)
    print('Original graph: n = {}, m = {}'.format(g.order(), g.size()))

    vrg_time = time()
    vrg_rules = []
    vrg_rules = extract_vrg(g, tree=[tree], lvl=0)

    print('VRG extracted in {} sec'.format(time() - vrg_time))
    print('#VRG rules: {}'.format(len(vrg_rules)))

    rule_dict, dedup_rules = deduplicate_rules(vrg_rules)  # rule_dict is dictionary keyed in by lhs

    graphs = []
    for i in range(num_graphs):
        gen_time = time()
        h = generate_graph(rule_dict)
        print('Generated graph #{}: n = {}, m = {}, time = {} sec'.format(i+1, h.order(), h.size(), time() - gen_time))
        graphs.append(h)

    print('total time: {} sec'.format(time() - tree_time))
    return graphs, vrg_rules


def main():
    """
    Driver method for VRG
    :return:
    """

    name = 'karate'  # 34    78
    # name = 'lesmis' # 77    254
    # name = 'football'  # 115   613
    # name = 'eucore'  # 1,005 25,571
    # name = 'bitcoin_alpha'  # 3,783 24,186
    # name = 'GrQc'  # 4,158 13,428
    # name = 'bitcoin_otc'  # 5,881 35,592
    # name = 'gnutella'  # 6,301 20,777
    # name = 'wikivote' # 7,115 103,689
    # name = 'hepth'  # 27,770 352,807
    # name = 'Enron'  # 36,692 183,831

    names = ['karate', 'lesmis', ]#'football', 'eucore', 'GrQc', 'bitcoin_alpha']
    k = 4
    # # g = get_graph('./tmp/{}.g'.format(names[0]))
    # g = get_graph()
    # g_original = nx.Graph(g)
    #
    # for _ in range(100):
    #     tree = partitions.approx_min_conductance_partitioning(g, k)  # consider Pickling the tree?
    #     extract_and_generate(g=g, k=4, tree=tree, mode='NO')
    #     g = nx.Graph(g_original)
    # return

    for name in names:
        # g = get_graph()
        g = get_graph('./tmp/{}.g'.format(name))
        g.name = name

        g_original = nx.Graph(g)
        k = 4

        # tree = partitions.n2v_partition(g)
        tree = partitions.approx_min_conductance_partitioning(g, k)  # consider Pickling the tree?
        tree_copy = deepcopy(tree)
        # print(tree)

        graph_dict = {}

        for mode in ['FULL', 'PART', 'NO']:
            graphs, vrg_rules = extract_and_generate(g=g, k=k, tree=tree, mode=mode, num_graphs=10)
            # analysis.analyze_rules(vrg_rules, mode)
            graph_dict[mode] = graphs
            g = deepcopy(g_original)
            tree = deepcopy(tree_copy)

        # count = 1
        # for g_full, g_part, g_no in zip(graph_dict['FULL'], graph_dict['PART'], graph_dict['NO']):
        #     analysis.compare_graphs(g_original, g_full, g_part, g_no, count)
        #     count += 1
    # plt.title('Level-wise cumulative MDL')
    # plt.xlabel('Level of discovery')
    # plt.ylabel('MDL')
    # plt.legend(loc='best')
    # plt.show()


if __name__ == '__main__':
    main()