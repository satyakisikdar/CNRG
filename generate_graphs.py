import networkx as nx
import os
import pickle
from time import time
import math
import logging
from joblib import Parallel, delayed
import csv
import sys
from typing import *

from src.VRG import VRG
from src.generate import generate_graph
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
    list_of_list_pickle = f'./{outdir}/{clustering}_list.pkl'
    # tree_pickle = f'./{outdir}/{clustering}_tree.pkl'
    # if os.path.exists()
    if not os.path.exists(f'./{outdir}'):
        os.makedirs(f'./{outdir}')

    if os.path.exists(list_of_list_pickle):
        print('Using existing pickle for {} clustering\n'.format(clustering))
        list_of_list_clusters = pickle.load(open(list_of_list_pickle, 'rb'))
    else:
        print('Running {} clustering...'.format(clustering))
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
        pickle.dump(list_of_list_clusters, open(list_of_list_pickle, 'wb'))
    return list_of_list_clusters

logging.basicConfig(level=logging.WARNING, format="%(message)s")

def make_dirs(outdir: str, name: str) -> None:
    """
    Make the necessary directories
    :param outdir:
    :param name:
    :return:
    """
    subdirs = ('grammars', 'graphs', 'rule_orders', 'trees', 'grammar_stats', 'gen_stats')

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


def dump_graphs(name: str, clustering: str, grammar_type: str) -> None:
    """
    Dump the stats
    :return:
    """
    original_graph = get_graph(name)
    outdir = 'dumps'
    make_dirs(outdir, name)  # make the directories if needed

    mus = range(2, min(original_graph.order(), 11))

    grammar_types = ('mu_random', 'mu_level', 'mu_dl', 'mu_level_dl', 'local_dl', 'global_dl')
    assert grammar_type in grammar_types, f'Invalid grammar type: {grammar_type}'

    # g_copy = original_graph.copy()

    num_graphs = 10
    # fieldnames = ('name', 'n', 'm', 'g_dl', 'type', 'mu', 'clustering', '#rules', 'grammar_dl', 'time')

    base_filename = f'{outdir}/grammars/{name}'
    for mu in mus:
        grammar_filename = f'{base_filename}/{clustering}_{grammar_type}_{mu}.pkl'
        graphs_filename = f'{outdir}/graphs/{name}/{clustering}_{grammar_type}_{mu}_graphs.pkl'
        rule_orders_filename = f'{outdir}/rule_orders/{name}/{clustering}_{grammar_type}_{mu}_orders.pkl'

        if not os.path.exists(grammar_filename):
            print('Grammar not found:', grammar_filename)
            continue

        if os.path.exists(graphs_filename):
            print('Graphs already generated')
            continue

        grammar = pickle.load(open(grammar_filename, 'rb'))

        graphs: List[LightMultiGraph] = []
        rule_orderings: List[List[int]] = []

        for i in range(num_graphs):
            rule_dict = dict(grammar.rule_dict)
            new_graph, rule_ordering = generate_graph(rule_dict)
            print(f'{name} {grammar_type}: {i+1}, n = {new_graph.order():_d} m = {new_graph.size():_d}')
            graphs.append(new_graph)
            rule_orderings.append(rule_ordering)


        pickle.dump(graphs, open(graphs_filename, 'wb'))
        pickle.dump(rule_orderings, open(rule_orders_filename, 'wb'))
    return


def generate_graphs(grammar: VRG, num_graphs=10):
    """

    :param grammar: VRG grammar object
    :param num_graphs: number of graphs
    :return: list of generated graphs and the rule orderings for each of the graphs
    """
    graphs: List[LightMultiGraph] = []
    rule_orderings: List[List[int]] = []

    for _ in range(num_graphs):
        rule_dict = dict(grammar.rule_dict)
        new_graph, rule_ordering = generate_graph(rule_dict)
        graphs.append(new_graph)
        rule_orderings.append(rule_ordering)
        # print(f'graph #{_ + 1} n = {new_graph.order()} m = {new_graph.size()} {rule_ordering}')

    return graphs, rule_orderings

def main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = 'sample'

    grammar_types = ('mu_random', 'mu_level', 'mu_dl', 'mu_level_dl', 'local_dl', 'global_dl')
    clustering_algs = ('cond', 'leiden', 'louvain', 'spectral', 'random')

    # get_grammar_parallel(name, 'leiden', 'mu_random')

    # gram = pickle.load(open('./dumps/grammars/karate/leiden_mu_random_3.pkl', 'rb'))
    # print(gram.rule_list)
    # outdir = 'dumps'
    # fieldnames = ('name', 'n', 'm', 'g_dl', 'type', 'mu', 'clustering', '#rules', 'grammar_dl', 'time')
    #
    # make_dirs(outdir, name)  # make the directories if needed

    # stats_path = f'{outdir}/gen_stats/{name}.csv'
    # mode = 'w'
    # if os.path.exists(stats_path):
    #     mode = 'a'
    #
    # if mode == 'w':
    #     with open(stats_path, mode) as csv_file:
    #         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #         writer.writeheader()

    Parallel(n_jobs=15, verbose=1)(delayed(dump_graphs)(name=name, clustering=clustering, grammar_type=grammar_type)
                                  for grammar_type in grammar_types
                                   for clustering in clustering_algs)


if __name__ == '__main__':
    main()
