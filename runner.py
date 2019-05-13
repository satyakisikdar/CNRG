import networkx as nx
import os
import pickle
from time import time
import math
import logging
from joblib import Parallel, delayed
from copy import deepcopy
import csv
from sys import argv

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


def get_clustering(g, outdir, clustering) -> TreeNode:
    '''
    wrapper method for getting dendrogram. uses an existing pickle if it can.
    :param g: graph
    :param outdir: output directory where picles are stored
    :param clustering: name of clustering method
    :return: root node of the dendrogram
    '''
    tree_pickle = f'./{outdir}/{clustering}_tree.pkl'
    if not os.path.exists(f'./{outdir}'):
        os.makedirs(f'./{outdir}')

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
        print(f'{clustering} clustering ran in {round(end_time, 3)} secs.')
    return root

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


def dump_grammar(name: str, clustering: str, grammar_type: str) -> None:
    """
    Dump the stats
    :return:
    """
    original_graph = get_graph(name)
    outdir = 'dumps'
    make_dirs(outdir, name)  # make the directories if needed

    mus = range(2, min(original_graph.order() // 2, 11))

    grammar_types = ('mu_random', 'mu_level', 'mu_dl', 'mu_level_dl', 'local_dl', 'global_dl')
    assert grammar_type in grammar_types, f'Invalid grammar type: {grammar_type}'

    g_copy = original_graph.copy()

    orig_root = get_clustering(g=g_copy, outdir=f'{outdir}/trees/{name}', clustering=clustering)

    fieldnames = ('name', 'n', 'm', 'g_dl', 'type', 'mu', 'clustering', '#rules', 'grammar_dl', 'time')

    g_dl = graph_dl(original_graph)

    base_filename = f'{outdir}/grammars/{name}'
    for mu in mus:
        orig_grammar = VRG(clustering=clustering, type=grammar_type, name=name, mu=mu)

        grammar = orig_grammar.copy()
        grammar_filename = f'{base_filename}/{grammar.clustering}_{grammar.type}_{grammar.mu}.pkl'

        if not os.path.exists(grammar_filename):  # the grammar does not exist
            g = original_graph.copy()
            root = deepcopy(orig_root)

            start_time = time()
            if 'mu' in grammar_type:
                extractor = MuExtractor(g=g, type=grammar.type, grammar=grammar, mu=mu, root=root)

            elif 'local' in grammar_type:
                extractor = LocalExtractor(g=g, type=grammar_type, grammar=grammar, mu=mu, root=root)

            else:
                assert grammar_type == 'global_dl', f'improper grammar type {grammar_type}'
                extractor = GlobalExtractor(g=g, type=grammar.type, grammar=grammar, mu=mu, root=root)

            extractor.generate_grammar()
            time_taken = round(time() - start_time, 3)

            grammar = extractor.grammar
            pickle.dump(grammar, open(grammar_filename, 'wb'))
        else:
            grammar = pickle.load(open(grammar_filename, 'rb'))
            time_taken = ''

        row = {'name': name, 'n': original_graph.order(), 'm': original_graph.size(), 'g_dl': g_dl,
               'type': grammar_type, 'mu': mu, 'clustering': clustering, '#rules': len(grammar), 'grammar_dl': grammar.cost,
               'time': time_taken}

        with open(f'{outdir}/grammar_stats/{name}.csv', 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(row)
    return


def main():
    if len(argv) > 1:
        name = argv[1]
    else:
        name = 'karate'

    grammar_types = ('mu_random', 'mu_level', 'mu_dl', 'mu_level_dl', 'local_dl', 'global_dl')
    clustering_algs = ('cond', 'leiden', 'louvain', 'spectral', 'random')

    # get_grammar_parallel(name, 'leiden', 'mu_random')

    # gram = pickle.load(open('./dumps/grammars/karate/leiden_mu_random_3.pkl', 'rb'))
    # print(gram.rule_list)
    outdir = 'dumps'
    fieldnames = ('name', 'n', 'm', 'g_dl', 'type', 'mu', 'clustering', '#rules', 'grammar_dl', 'time')

    make_dirs(outdir, name)  # make the directories if needed

    with open(f'{outdir}/grammar_stats/{name}.csv', 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    Parallel(n_jobs=15, verbose=1)(delayed(dump_grammar)(name=name, clustering=clustering, grammar_type=grammar_type)
                                  for grammar_type in grammar_types
                                   for clustering in clustering_algs)


if __name__ == '__main__':
    main()
    # par_main()
