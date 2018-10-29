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
import math
import multiprocessing as mp

sys.path.extend([os.getcwd(), os.path.dirname(os.getcwd())])

import vrgs.partitions as partitions
from vrgs.funky_extract import funky_extract
from vrgs.Tree import create_tree
import vrgs.analysis as analysis
from vrgs.MDL import graph_mdl
from vrgs.other_graph_generators import chung_lu, kronecker2_random_graph, bter_wrapper, vog, subdue, hrg_wrapper
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
        name = g.name
        g = nx.convert_node_labels_to_integers(g)
        g.name = name
    return g


def write_graph_stats(g, name):
    fieldnames = ('name', 'selection', 'clustering', 'k', 'count',
                  'actual_n', 'actual_m', 'actual_MDL',
                  'full_n', 'full_m', 'full_rules', 'full_MDL', 'GCD_full', 'CVM_full_d', 'CVM_full_pr',
                  'part_n', 'part_m', 'part_rules', 'part_MDL', 'GCD_part', 'CVM_part_d', 'CVM_part_pr',
                  'no_n', 'no_m', 'no_rules', 'no_MDL', 'GCD_no', 'CVM_no_d', 'CVM_no_pr')

    stats_filename = './stats/{}_new_stats.csv'.format(name)
    with open(stats_filename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    orig_graph = deepcopy(g)
    g_mdl = graph_mdl(orig_graph)

    for k in range(4, 6):
        for selection in ('random', 'mdl', 'level', 'level_mdl'):
            for clustering in ('random', 'louvain', 'cond', 'spectral', 'node2vec'):
                if clustering == 'random':
                    tree = partitions.get_random_partition(g)
                elif clustering == 'louvain':
                    tree = partitions.louvain(g)
                elif clustering == 'cond':
                    tree = partitions.approx_min_conductance_partitioning(g)
                elif clustering == 'spectral':
                    tree = partitions.spectral_kmeans(g, K=int(math.sqrt(g.order() // 2)))
                else:
                    tree = partitions.get_node2vec(g)

                root = create_tree(tree)
                new_graph_count = 2  # number of graphs generated

                print('\n{} {} {} {}'.format(name, selection, clustering, k))


                row = {'name': name, 'selection': selection, 'clustering': clustering, 'k': k,
                       'actual_n': orig_graph.order(), 'actual_m': orig_graph.size(), 'actual_MDL': g_mdl}
                rows = [dict() for _ in range(new_graph_count)]  # for each of the row, this info is the same

                for mode in ('full', 'part', 'no'):
                    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode,
                                            selection=selection, clustering=clustering)

                    graphs = grammar.generate_graphs(count=new_graph_count)
                    row['{}_rules'.format(mode)] = len(grammar)
                    row['{}_MDL'.format(mode)] = grammar.get_cost()

                    for i in range(new_graph_count):
                        rows[i] = {**row, **rows[i]}

                    for count, g1 in enumerate(graphs):
                        rows[count]['count'] = count + 1
                        rows[count]['{}_n'.format(mode)] = g1.order()
                        rows[count]['{}_m'.format(mode)] = g1.size()

                        gcd, cvm_deg, cvm_page = analysis.compare_two_graphs(orig_graph, g1)
                        rows[count]['GCD_{}'.format(mode)] = gcd
                        rows[count]['CVM_{}_d'.format(mode)] = cvm_deg
                        rows[count]['CVM_{}_pr'.format(mode)] = cvm_page


                with open(stats_filename, 'a') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    for row in rows:
                        writer.writerow(row)
                    writer.writerow({})  # a blank row after each batch

            with open(stats_filename, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({})  # two blank rows after each k



def main():
    """
    Driver method for VRG
    :return:
    """
    np.seterr(all='ignore')

    names = ('karate', 'lesmis', 'dolphins', 'football', 'eucore', 'GrQc', 'gnutella', 'wikivote')

    for name in names[:1]:
        g = get_graph('./tmp/{}.g'.format(name))
        g.name = name
        write_graph_stats(g, name)

    return

    g = get_graph('./tmp/football.g')
    # g = get_graph()
    # g = nx.convert_node_labels_to_integers(g, first_label=0)

    # tree = partitions.approx_min_conductance_partitioning(g)
    # tree = partitions.spectral_kmeans(g, K=4)
    tree = partitions.louvain(g)
    # tree = partitions.get_random_partition(g, seed=10)
    # print(tree)
    # tree = [[[[6], [7]], [[8], [9]]], [[[5], [[3], [4]]], [[1], [2]]]]


    root = create_tree(tree)
    k = 4
    mode = 'full'

    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection='random', clustering=clustering)
    print(grammar)

    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection='mdl', clustering=clustering)
    print(grammar)

    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection='level', clustering=clustering)
    print(grammar)

    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection='level_mdl', clustering=clustering)
    print(grammar)

    mode = 'part'

    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection='random', clustering=clustering)
    print(grammar)

    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection='mdl', clustering=clustering)
    print(grammar)

    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection='level', clustering=clustering)
    print(grammar)

    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection='level_mdl', clustering=clustering)
    print(grammar)

    mode = 'no'

    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection='random', clustering=clustering)
    print(grammar)

    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection='mdl', clustering=clustering)
    print(grammar)

    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection='level', clustering=clustering)
    print(grammar)

    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection='level_mdl', clustering=clustering)
    print(grammar)

    return

    root = create_tree(tree)
    mode = 'full'
    k  = 3

    grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection=False)
    print(grammar.get_cost())

    return
    # networkx to matrix
    # np.savetxt('karate.mat', nx.to_numpy_matrix(g), fmt='%d')

    name = 'karate'
    g = get_graph('./tmp/{}.g'.format(name))
    g.name = name

    # g = get_graph()
    # g_hrgs = hrg_wrapper(g, n=5)
    # return
    #
    # g_chung_lu = chung_lu(g)
    # print('n: {} m: {} n: {}, m: {} GCD {}'.format(g.order(), g.size(), g_chung_lu.order(), g_chung_lu.size(),
    #                                              GCD(g, g_chung_lu)))
    #
    # k = int(math.log(g.size(), 2))
    #
    # g_kron = kronecker2_random_graph(k, [[0.99, 0.6683], [0.6683, 0.025]], directed=False)
    #
    # print('n: {} m: {} n: {}, m: {} GCD {}'.format(g.order(), g.size(), g_kron.order(), g_kron.size(),
    #                                              GCD(g, g_kron)))

    structures = subdue(g)
    #
    # print('{} structures discovered by SUBDUE'.format(len(structures)))

    # structures = vog(g)
    #
    # print('{} structures discovered by VoG'.format(len(structures)))
    # vog_mdl = 0
    # for struct in structures:
    #     vog_mdl += graph_mdl(struct)
    # print('VoG MDL', vog_mdl, 'bits')

    # structures = vog(g)

    print('{} structures discovered by VoG'.format(len(structures)))

    start = time()
    pool =  mp.Pool(processes=5)
    for struct in structures:
        pool.apply_async(graph_mdl, args=(struct[0],), callback=mdl_callback)
    pool.close()
    pool.join()
    print('VoG MDL', mdl_sum, 'bits', round(time() - start, 3), 'secs')

    return

    g_bter = bter_wrapper(g)
    print('n: {} m: {} n: {}, m: {} GCD {}'.format(g.order(), g.size(), g_bter.order(), g_bter.size(),
                                                 GCD(g, g_bter)))

    return

    orig_graph = deepcopy(g)
    tree = partitions.approx_min_conductance_partitioning(g)
    # tree = [[[[6], [7]], [[8], [9]]], [[[5], [[3], [4]]], [[1], [2]]]]

    root = create_tree(tree)
    new_graph_count = 2  # number of graphs generated

    k = 4

    for mode in ('full', 'part', 'no'):
        grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode, selection=True)
        print('\n"{}" original graph {} nodes {} edges'.format(mode.upper(), orig_graph.order(),
                                                               orig_graph.size()))

        graphs = grammar.generate_graphs(count=new_graph_count)



if __name__ == '__main__':
    main()
