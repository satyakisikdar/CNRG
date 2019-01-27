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
from copy import deepcopy
import os
import sys
import numpy as np
import math
import pickle
import argparse
import glob

sys.path.extend([os.getcwd(), os.path.dirname(os.getcwd()), './src'])

import src.partitions as partitions
from src.extract import extract
from src.Tree import create_tree


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


def parse_args():
    graph_names = [fname[: fname.find('.g')].split('/')[-1]
                   for fname in glob.glob('./src/tmp/*.g')]
    clustering_algs = ['louvain', 'spectral', 'cond', 'node2vec', 'random', 'leiden']
    selections = ['random', 'level', 'level_mdl', 'mdl']

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # formatter class shows defaults in help

    # using choices we can control the inputs. metavar='' prevents printing the choices in the help preventing clutter
    parser.add_argument('-g', '--graph', help='Name of the graph', default='karate', choices=graph_names, metavar='')
    parser.add_argument('-c', '--clustering', help='Clustering method to use', default='louvain',
                        choices=clustering_algs, metavar='')
    parser.add_argument('-b', '--boundary', help='Degree of boundary information to store', default='part',
                        choices=['full', 'part', 'no'])
    parser.add_argument('-l', '--lamb', help='Size of RHS (lambda)', default=5, type=int)
    parser.add_argument('-s', '--selection', help='Selection strategy', default='level', choices=selections, metavar='')
    parser.add_argument('-o', '--outdir', help='Name of the output directory', default='output')
    parser.add_argument('-n', help='Number of graphs to generate', default=5, type=int)
    return parser.parse_args()


def get_clustering(g, name, outdir, clustering):
    '''
    wrapper method for getting dendrogram. uses an existing pickle if it can.
    :param g: graph
    :param name: name of the graph
    :param outdir: output directory where picles are stored
    :param clustering: name of clustering method
    :return: root node of the dendrogram
    '''
    tree_pickle = './{}/{}_{}_tree.pkl'.format(outdir, name, clustering)
    if os.path.exists(tree_pickle):
        print('Using existing pickle for {} clustering\n'.format(clustering))
        root = pickle.load(open(tree_pickle, 'rb'))
    else:
        print('Running {} clustering...'.format(clustering), end='\r')
        start_time = time()
        if clustering == 'random':
            list_of_list_clusters = partitions.get_random_partition(g)
        elif clustering == 'leiden':
            list_of_list_clusters = partitions.leiden(g)
        elif clustering == 'louvain':
            list_of_list_clusters = partitions.louvain(g)
        elif clustering == 'cond':
            list_of_list_clusters = partitions.approx_min_conductance_partitioning(g)
        elif clustering == 'spectral':
            list_of_list_clusters = partitions.spectral_kmeans(g, K=int(math.sqrt(g.order() // 2)))
        else:
            list_of_list_clusters = partitions.get_node2vec(g)
        root = create_tree(list_of_list_clusters)
        end_time = time() - start_time

        pickle.dump(root, open(tree_pickle, 'wb'))
        print('{} clustering ran in {} secs. Pickled as a Tree object.\n'.format(clustering, round(end_time, 3)))
    return root


def get_grammar(g, name, outdir, clustering, selection, k, mode, root):
    '''
    returns a VRG object of type 'mode'
    :param g: graph
    :param name: name of the graph
    :param outdir: output directory
    :param clustering: name of clustering method
    :param selection: selection algorithm used
    :param k: lambda
    :param mode: full, part, or no boundary info
    :param root: root of the dendrogram
    :return: VRG object
    '''
    grammar_pickle = './{}/{}_{}_{}_{}_grammar.pkl'.format(outdir, name, clustering, selection, k)
    if os.path.exists(grammar_pickle):
        print('Using existing pickle for grammar: lambda: {}, boundary info: {}, selection: {}.\n'.format(k, mode, selection))
        grammar = pickle.load(open(grammar_pickle, 'rb'))
    else:
        print('Starting grammar induction: lambda: {}, boundary info: {}, selection: {}...'.format(k, mode, selection), end='\r')
        start_time = time()
        grammar = extract(g=deepcopy(g), root=deepcopy(root), k=k, mode=mode,
                          selection=selection, clustering=clustering)
        end_time = time() - start_time
        pickle.dump(grammar, open(grammar_pickle, 'wb'))
        print('Grammar: lambda: {}, boundary info: {}, selection: {}. Generated in {} secs. Pickled as a VRG object.\n'.format(k, mode, selection, round(end_time, 3)))
    return grammar

def main():
    """
    Driver method for VRG
    :return:
    """
    args = parse_args()

    name, clustering, mode, k, selection, outdir = args.graph, args.clustering, args.boundary, args.lamb,\
                                                   args.selection, args.outdir

    if not os.path.isdir('./{}'.format(outdir)):  # make the output directory if it doesn't exist already
        os.mkdir(outdir)

    print('Reading graph "{}"...'.format(name), end='\r')  # using \r allows to rewrite the current line
    start_time = time()
    g = get_graph('./src/tmp/{}.g'.format(name))
    end_time = time() - start_time
    g.name = name

    print('Graph "{}", n: {}, m: {}, read in {} secs\n'.format(name, g.order(), g.size(), round(end_time, 3)))

    root = get_clustering(g=g, name=name, outdir=outdir, clustering=clustering)

    grammar = get_grammar(g=g, name=name, outdir=outdir, clustering=clustering, k=k, mode=mode, selection=selection,
                          root=root)

    print('Generating {} graphs...'.format(args.n), end='\r')
    start_time = time()
    graphs = grammar.generate_graphs(count=args.n)
    end_time = time() - start_time

    pickle.dump(graphs, open('./{}/{}_{}_{}_{}_graphs.pkl'.format(outdir, name, clustering, selection, k), 'wb'))
    print('{} graphs generated in {} secs. Pickled as a list of nx.Graph objects.'.format(args.n, round(end_time, 3)))


if __name__ == '__main__':
    np.seterr(all='ignore')
    main()
