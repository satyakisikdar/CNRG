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

import vrgs.globals as globals
import vrgs.partitions as partitions
import vrgs.full_info as full_info
import vrgs.part_info as part_info

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

    return rule_dict


def main():
    """
    Driver method for VRG
    :return:
    """
    # g = get_graph()
    # g = get_graph('BA')
    # g = get_graph('./tmp/karate.g')           # 34    78
    # g = get_graph('./tmp/lesmis.g')           # 77    254
    # g = get_graph('./tmp/football.g')         # 115   613
    g = get_graph('./tmp/eucore.g')           # 1,005 25,571
    # g = get_graph('./tmp/bitcoin_alpha.g')    # 3,783 24,186
    # g = get_graph('./tmp/GrQc.g')             # 5,242 14,496
    # g = get_graph('./tmp/bitcoin_otc.g')      # 5,881 35,592
    # g = get_graph('./tmp/gnutella.g')         # 6,301 20,777
    # g = get_graph('./tmp/wikivote.g')         # 7,115 103,689
    # g = get_graph('./tmp/hepth.g')            # 27,770 352,807
    # g = get_graph('./tmp/Enron.g')            # 36,692 183,831


    globals.original_graph = g.copy()
    tree_time = time()

    k = 4
    print('k =', k)
    print('n = {}, m = {}'.format(g.order(), g.size()))

    # tree = partitions.n2v_partition(g)
    tree = partitions.approx_min_conductance_partitioning(g, k)
    # print(tree)
    print('tree done in {} sec!'.format(time() - tree_time))

    vrg_time = time()
    # vrg_rules = full_info.extract_vrg(g, tree=[tree], lvl=0)  # root is at level 0
    vrg_rules = part_info.extract_vrg(g, tree=[tree], lvl=0)

    print('VRG extracted in {} sec'.format(time() - vrg_time))
    print('#VRG rules: {}'.format(len(vrg_rules)))

    rule_dict = deduplicate_rules(vrg_rules)  # rule_dict is dictionary keyed in by lhs
    # print(uniq_rules)

    error_count = 0
    for i in range(5):
        gen_time = time()
        # h = full_info.generate_graph(rule_dict)
        h = part_info.generate_graph(rule_dict)
        if h == -1:
            error_count += 1
        else:
            print('{}) n = {}, m = {}, time = {} sec'.format(i+1, h.order(), h.size(), time() - gen_time))
    print('{} errors'.format(error_count))
    print('total time: {} sec'.format(time() - tree_time))
    return


if __name__ == '__main__':
    main()