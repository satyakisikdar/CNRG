import networkx as nx
import os
import pickle
from time import time
import math
import logging
from joblib import Parallel, delayed
from copy import deepcopy
import argparse
import csv
import sys
import glob
from tqdm import tqdm
from shutil import copyfile

sys.setrecursionlimit(1_000_000)

from src.VRG import VRG
from src.extract import MuExtractor, LocalExtractor, GlobalExtractor
from src.Tree import create_tree, TreeNode
import src.partitions as partitions
from src.LightMultiGraph import LightMultiGraph
from src.MDL import graph_dl

def get_graph(filename='sample') -> LightMultiGraph:
    start_time = time()
    if filename == 'sample':
        # g = nx.MultiGraph()
        g = nx.Graph()
        g.add_edges_from([(1, 2), (1, 3), (1, 5),
                          (2, 4), (2, 5), (2, 7),
                          (3, 4), (3, 5),
                          (4, 5), (4, 9),
                          (6, 7), (6, 8), (6, 9),
                          (7, 8), (7, 9),
                          (8, 9)])
    elif filename == 'BA':
        g = nx.barabasi_albert_graph(10, 2, seed=42)
        # g = nx.MultiGraph(g)
        g = nx.Graph()
    else:
        g = nx.read_edgelist(f'./src/tmp/{filename}.g', nodetype=int, create_using=nx.Graph())
        # g = nx.MultiGraph(g)
        if not nx.is_connected(g):
            g = max(nx.connected_component_subgraphs(g), key=len)
        name = g.name
        g = nx.convert_node_labels_to_integers(g)
        g.name = name

    g_new = LightMultiGraph()
    g_new.add_edges_from(g.edges())

    end_time = time() - start_time
    print(f'Graph: {filename}, n = {g.order():_d}, m = {g.size():_d} read in {round(end_time, 3):_g}s.')

    return g_new


def get_clustering(g, outdir, clustering):
    '''
    wrapper method for getting dendrogram. uses an existing pickle if it can.
    :param g: graph
    :param outdir: output directory where picles are stored
    :param clustering: name of clustering method
    :return: root node of the dendrogram
    '''
    # list_of_list_pickle = f'./{outdir}/{clustering}_list.pkl'
    # # tree_pickle = f'./{outdir}/{clustering}_tree.pkl'
    # # if os.path.exists()
    # if not os.path.exists(f'./{outdir}'):
    #     os.makedirs(f'./{outdir}')
    #
    # if os.path.exists(list_of_list_pickle):
    #     print('Using existing pickle for {} clustering\n'.format(clustering))
    #     list_of_list_clusters = pickle.load(open(list_of_list_pickle, 'rb'))
    # else:
    tqdm.write('Running {} clustering...'.format(clustering))
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
        # pickle.dump(list_of_list_clusters, open(list_of_list_pickle, 'wb'))
    return list_of_list_clusters

logging.basicConfig(level=logging.WARNING, format="%(message)s")

def make_dirs(outdir: str, name: str) -> None:
    """
    Make the necessary directories
    :param outdir:
    :param name:
    :return:
    """
    subdirs = ('grammars', 'graphs', 'rule_orders', 'trees', 'grammar_stats')

    for dir in subdirs:
        dir_path = f'./{outdir}/{dir}/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if dir == 'grammar_stats':
            continue
        dir_path += f'{name}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    return


def dump_grammar(name: str, clustering: str, grammar_type: str, mu: int) -> None:
    """
    Dump the stats
    :return:
    """
    original_graph = get_graph(name)
    outdir = 'dumps'
    make_dirs(outdir, name)  # make the directories if needed

    grammar_types = ('mu_random', 'mu_level', 'mu_dl', 'mu_level_dl', 'local_dl', 'global_dl')
    assert grammar_type in grammar_types, f'Invalid grammar type: {grammar_type}'

    g_copy = original_graph.copy()

    list_of_list_clusters = get_clustering(g=g_copy, outdir=f'{outdir}/trees/{name}', clustering=clustering)

    g_dl = graph_dl(original_graph)

    grammar = VRG(clustering=clustering, type=grammar_type, name=name, mu=mu)

    g = original_graph.copy()
    list_of_list_clusters_copy = list_of_list_clusters[: ]
    root = create_tree(list_of_list_clusters_copy)
    start_time = time()
    if 'mu' in grammar_type:
        extractor = MuExtractor(g=g, type=grammar.type, grammar=grammar, mu=mu, root=root)

    elif 'local' in grammar_type:
        extractor = LocalExtractor(g=g, type=grammar_type, grammar=grammar, mu=mu, root=root)

    else:
        assert grammar_type == 'global_dl', f'improper grammar type {grammar_type}'
        extractor = GlobalExtractor(g=g, type=grammar.type, grammar=grammar, mu=mu, root=root)

    extractor.generate_grammar()
    time_taken = round(time() - start_time, 4)

    grammar = extractor.grammar

    row = {'name': name, 'n': original_graph.order(), 'm': original_graph.size(), 'g_dl': round(g_dl, 3),
           'type': grammar_type, 'mu': mu, 'clustering': clustering, '#rules': len(grammar), 'grammar_dl': round(grammar.cost, 3),
           'time': time_taken, 'compression': round(grammar.cost / g_dl, 3)}

    # tqdm.write(f"name: {name}, n: {row['n']}, m: {row['m']}, mu: {row['mu']}, graph_dl: {g_dl}, grammar_dl: {grammar.cost},"
    #            f"compression: {row['compression']}, time: {time_taken}s")
    tqdm.write(f"name: {name}, original: {g_dl}, grammar: {grammar.cost}, time: {time_taken}")
    return


def old_main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = 'sample'

    grammar_types = ('mu_random', 'mu_level', 'mu_dl', 'mu_level_dl', 'local_dl', 'global_dl')
    clustering_algs = ('cond', 'leiden', 'louvain', 'spectral', 'random')

    # get_grammar_parallel(name, 'leiden', 'mu_random')

    # gram = pickle.load(open('./dumps/grammars/karate/leiden_mu_random_3.pkl', 'rb'))
    # print(gram.rule_list)
    outdir = 'dumps'
    fieldnames = ('name', 'n', 'm', 'g_dl', 'type', 'mu', 'clustering', '#rules', 'grammar_dl', 'time')

    make_dirs(outdir, name)  # make the directories if needed

    stats_path = f'{outdir}/grammar_stats/{name}.csv'
    mode = 'w'
    if os.path.exists(stats_path):
        print('stats file exists')
        mode = 'a'

    if mode == 'w':
        with open(stats_path, mode) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    Parallel(n_jobs=15, verbose=1)(delayed(dump_grammar)(name=name, clustering=clustering, grammar_type=grammar_type)
                                  for grammar_type in grammar_types
                                   for clustering in clustering_algs)


def parse_args():
    graph_names = [fname[: fname.find('.g')].split('/')[-1]
                   for fname in glob.glob('./src/tmp/*.g')]
    clustering_algs = ['leiden', 'louvain', 'spectral', 'cond', 'node2vec', 'random']
    grammar_types = ('mu_random', 'mu_level', 'mu_dl', 'mu_level_dl', 'local_dl', 'global_dl')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # formatter class shows defaults in help

    # using choices we can control the inputs. metavar='' prevents printing the choices in the help preventing clutter
    parser.add_argument('-g', '--graph', help='Name of the graph', default='karate', choices=graph_names,
                        metavar='')

    parser.add_argument('-c', '--clustering', help='Clustering method to use', default='leiden',
                        choices=clustering_algs, metavar='')

    parser.add_argument('-b', '--boundary', help='Degree of boundary information to store', default='part',
                        choices=['full', 'part', 'no'])

    parser.add_argument('-m', '--mu', help='Size of RHS (mu)', default=4, type=int)

    parser.add_argument('-t', '--type', help='Grammar type', default='mu_level_dl', choices=grammar_types, metavar='')

    parser.add_argument('-o', '--outdir', help='Name of the output directory', default='output')

    parser.add_argument('-n', help='Number of graphs to generate', default=5, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    name, clustering, mode, mu, type, outdir = args.graph, args.clustering, args.boundary, args.mu, \
                                                   args.type, args.outdir

    dump_grammar(name=name, grammar_type=type, clustering=clustering, mu=mu)


if __name__ == '__main__':
    main()
