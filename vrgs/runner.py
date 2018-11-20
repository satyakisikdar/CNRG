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
import pickle

sys.path.extend([os.getcwd(), os.path.dirname(os.getcwd())])

import vrgs.partitions as partitions
from vrgs.funky_extract import funky_extract
from vrgs.Tree import create_tree
import vrgs.analysis as analysis
from vrgs.MDL import graph_mdl, gamma_code
from vrgs.other_graph_generators import chung_lu_graphs, kronecker2_random_graph, bter_graphs, vog, subdue, hrg_wrapper
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


def write_graph_stats(g, name, write_flag):
    fieldnames = ('name', 'selection', 'clustering', 'k', 'count',
                  'actual_n', 'actual_m', 'actual_MDL',
                  'full_n', 'full_m', 'full_rules', 'full_MDL', 'GCD_full', 'CVM_full_d', 'CVM_full_pr',
                  'part_n', 'part_m', 'part_rules', 'part_MDL', 'GCD_part', 'CVM_part_d', 'CVM_part_pr',
                  'no_n', 'no_m', 'no_rules', 'no_MDL', 'GCD_no', 'CVM_no_d', 'CVM_no_pr')

    if write_flag:
        stats_filename = './new_stats/{}_stats.csv'.format(name)
        with open(stats_filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    orig_graph = deepcopy(g)
    g_mdl = graph_mdl(orig_graph)
    true_deg = list(orig_graph.degree().values())
    true_page = list(map(lambda x: round(x, 3), nx.pagerank_scipy(orig_graph).values()))



    for clustering in ('louvain', 'cond', 'spectral', 'node2vec', 'random'):
        print('clustering:', clustering)

        if os.path.isfile('./pickles/trees/{}_{}.tree'.format(name, clustering)):
            tree = pickle.load(open('./pickles/trees/{}_{}.tree'.format(name, clustering), 'rb'))
            print('read tree from pickle')
        else:
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
            pickle.dump(tree, open('./pickles/trees/{}_{}.tree'.format(name, clustering), 'wb'))

        root = create_tree(tree)

        for selection in ('random', 'mdl', 'level', 'level_mdl'):
            new_graph_count = 5  # number of graphs generated

            for k in range(3, 5):
                print('\nname:{} selection:{} clustering:{} k:{}'.format(name, selection, clustering, k))
                row = {'name': name, 'selection': selection, 'clustering': clustering, 'k': k,
                       'actual_n': orig_graph.order(), 'actual_m': orig_graph.size(), 'actual_MDL': g_mdl}

                rows = [dict() for _ in range(new_graph_count)]  # for each of the row, this info is the same
                for mode in ('full', 'part', 'no'):

                    if os.path.isfile('./pickles/grammars/{}_{}_{}_{}.grammar'.format(name, clustering, selection, k)):
                        grammar = pickle.load(open('./pickles/grammars/{}_{}_{}_{}.grammar'.format(name, clustering, selection, k), 'rb'))
                        print('read grammar from pickle')

                    else:
                        grammar = funky_extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode,
                                                selection=selection, clustering=clustering)

                        pickle.dump(grammar, open('./pickles/grammars/{}_{}_{}_{}.grammar'.format(name, clustering, selection, k), 'wb'))


                    graphs = []
                    for i in range(new_graph_count):
                        if os.path.isfile('./pickles/graphs/{}_{}_{}_{}_{}.g'.format(name, clustering, selection, k, i+1)):
                            graphs.append(nx.read_edgelist('./pickles/graphs/{}_{}_{}_{}_{}.g'.format(name, clustering, selection, k, i+1), nodetype=int))
                            print('reading pickled graph')

                    if len(graphs) != new_graph_count:
                        graphs = grammar.generate_graphs(count=new_graph_count)

                        [nx.write_edgelist(graph, './pickles/graphs/{}_{}_{}_{}_{}.g'.format(name, clustering, selection, k, i+1), data=False)
                            for i, graph in enumerate(graphs)]

                    row['{}_rules'.format(mode)] = len(grammar)
                    row['{}_MDL'.format(mode)] = grammar.get_cost()

                    for i in range(new_graph_count):
                        rows[i] = {**row, **rows[i]}

                    for count, g1 in enumerate(graphs):
                        g1.name = orig_graph.name
                        rows[count]['count'] = count + 1
                        rows[count]['{}_n'.format(mode)] = g1.order()
                        rows[count]['{}_m'.format(mode)] = g1.size()

                        gcd, cvm_deg, cvm_page = analysis.compare_two_graphs(orig_graph, g1, true_deg=true_deg,
                                                                             true_page=true_page)
                        rows[count]['GCD_{}'.format(mode)] = gcd
                        rows[count]['CVM_{}_d'.format(mode)] = cvm_deg
                        rows[count]['CVM_{}_pr'.format(mode)] = cvm_page

                if write_flag:
                    with open(stats_filename, 'a') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        for row in rows:
                            writer.writerow(row)
                        writer.writerow({})  # a blank row after each batch

            if write_flag:
                with open(stats_filename, 'a') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({})  # two blank rows after each k


def write_edgelists(graphs, name, gname):
    for i, g in enumerate(graphs):
        nx.write_edgelist(g, './graph_comp/{}/{}_{}.g'.format(name, gname, i+1), data=False)

def graph_generation(g):
    name = g.name

    if not os.path.isfile('./graph_comp/hrg/{}_1.g'.format(name)):
        hrgs = hrg_wrapper(g, n=5)
        write_edgelists(hrgs, 'hrg', name)

    # bter = get_best(bter_graphs(g, n=5), g)
    # nx.write_edgelist(bter, './graph_comp/{}_bter.g'.format(name))

    # krons = kronecker2_random_graph()

    if not os.path.isfile('./graph_comp/cl/{}_1.g'.format(name)):
        chungs = chung_lu_graphs(g, n=5)
        write_edgelists(chungs, 'cl', name)






def main():
    """
    Driver method for VRG
    :return:
    """
    np.seterr(all='ignore')

    names = ('karate', 'lesmis', 'dolphins', 'football', 'eucore', 'GrQc', 'gnutella', 'wikivote')

    # if len(sys.argv) < 2 or sys.argv[1] not in names:
    #     print('enter name from', names)
    #     return
    # name = sys.argv[1]
    name = 'karate'
    g = get_graph('./tmp/{}.g'.format(name))
    g.name = name
    # graph_generation(g)

    write_graph_stats(g, name, write_flag=True)

    return

if __name__ == '__main__':
    main()
