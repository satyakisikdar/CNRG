import networkx as nx
from typing import Union, List
from time import time
import numpy as np
import logging
import os
import pickle
import sys
import csv
from tqdm import tqdm, trange

from numpy import linalg as la
from joblib import Parallel, delayed
from scipy.sparse import issparse
from scipy import sparse as sps

from src.LightMultiGraph import LightMultiGraph
from src.GCD import GCD

def cvm_distance(data1, data2):
    data1, data2 = map(np.asarray, (data1, data2))
    n1 = len(data1)
    n2 = len(data2)
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    d = np.sum(np.absolute(cdf1 - cdf2))
    return np.round(d / len(cdf1), 3)


def lambda_dist(g1, g2, k=None, p=2) -> float:
    """
    compare the euclidean distance between the top-k eigenvalues of the laplacian
    :param g1:
    :param g2:
    :param k:
    :param p:
    :return:
    """
    if k is None:
        k = min(g1.order(), g2.order())

    lambda_seq_1 = np.array(sorted(nx.linalg.laplacian_spectrum(g1), reverse=True)[: k])
    lambda_seq_2 = np.array(sorted(nx.linalg.laplacian_spectrum(g2), reverse=True)[: k])
    return round(la.norm(lambda_seq_1 - lambda_seq_2, ord=p) / k, 3)


def _pad(A,N):
    """Pad A so A.shape is (N,N)"""
    n,_ = A.shape
    if n>=N:
        return A
    else:
        if issparse(A):
            # thrown if we try to np.concatenate sparse matrices
            side = sps.csr_matrix((n,N-n))
            bottom = sps.csr_matrix((N-n,N))
            A_pad = sps.hstack([A,side])
            A_pad = sps.vstack([A_pad,bottom])
        else:
            side = np.zeros((n,N-n))
            bottom = np.zeros((N-n,N))
            A_pad = np.concatenate([A,side],axis=1)
            A_pad = np.concatenate([A_pad,bottom])
        return A_pad


def fast_bp(A,eps=None):
    n, m = A.shape
    degs = np.array(A.sum(axis=1)).flatten()
    if eps is None:
        eps = 1 / (1 + max(degs))
    I = sps.identity(n)
    D = sps.dia_matrix((degs,[0]),shape=(n,n))
    # form inverse of S and invert (slow!)
    Sinv = I + eps**2*D - eps*A
    try:
        S = la.inv(Sinv)
    except:
        Sinv = sps.csc_matrix(Sinv)
        S = sps.linalg.inv(Sinv)
    return S


def deltacon0(g1, g2, eps=None):
    n1, n2 = g1.order(), g2.order()
    N = max(n1, n2)
    A1, A2 = [_pad(A, N) for A in [nx.to_numpy_array(g1), nx.to_numpy_array(g2)]]
    S1, S2 = [fast_bp(A, eps=eps) for A in [A1, A2]]
    dist = np.abs(np.sqrt(S1) - np.sqrt(S2)).sum()
    return round(dist, 3)


def compare_two_graphs(g_true: nx.Graph, g_test: Union[nx.Graph, LightMultiGraph], true_deg=None, true_page=None):
    """
    Compares two graphs
    :param g_true: actual graph
    :param g_test: generated graph
    :return:
    """
    if true_deg is None:
        true_deg = nx.degree_histogram(g_true)

    if true_page is None:
        true_page = list(nx.pagerank_scipy(g_true).values())

    start = time()
    g_test_deg = nx.degree_histogram(g_test)
    deg_time = time() - start

    start = time()
    g_test_pr = list(nx.pagerank_scipy(g_test).values())
    page_time = time() - start

    start = time()
    gcd = GCD(g_true, g_test, 'orca')
    gcd_time = time() - start

    start = time()
    cvm_deg = cvm_distance(true_deg, g_test_deg)
    cvm_page = cvm_distance(true_page, g_test_pr)
    cvm_time = time() - start

    ld = lambda_dist(g_true, g_test, k=min(g_true.order(), g_test.order(), 10))

    dc0 = deltacon0(g_true, g_test)

    logging.debug(f'times: deg {round(deg_time, 3)}s, page {round(page_time, 3)}s, gcd {round(gcd_time, 3)}s, cvm {round(cvm_time, 3)}s')
    return gcd, cvm_deg, cvm_page, ld, dc0



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
    # tqdm.write(f'Graph: {filename}, n = {g.order():_d}, m = {g.size():_d} read in {round(end_time, 3):_g}s.')

    return g_new


logging.basicConfig(level=logging.DEBUG, format="%(message)s")

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


def dump_graph_stats(name: str, clustering: str, grammar_type: str) -> None:
    """
    Dump the stats
    :return:
    """
    g_true = get_graph(name)
    g_true.name = f'{name}_{clustering}_{grammar_type}_true'

    true_deg = nx.degree_histogram(g_true)
    true_pr = list(nx.pagerank_scipy(g_true).values())


    outdir = 'dumps'
    make_dirs(outdir, name)  # make the directories if needed
    mus = range(2, min(g_true.order(), 11))

    grammar_types = ('mu_random', 'mu_level', 'mu_dl', 'mu_level_dl', 'local_dl', 'global_dl')
    assert grammar_type in grammar_types, f'Invalid grammar type: {grammar_type}'

    base_filename = f'{outdir}'  # /gen_stats/{name}'

    fieldnames = ('name', 'orig_n', 'orig_m', 'type', 'mu', 'clustering', 'i', 'gen_n', 'gen_m', '#comps',
                  'gcd', 'deltacon0', 'lambda_dist', 'cvm_deg', 'cvm_pr')

    stats_filename = f'{base_filename}/gen_stats/{name}.csv'

    tqdm.write(f'\nanalyzing {name}_{clustering}_{grammar_type}')
    for mu in trange(2, min(g_true.order(), 11)):
        graphs_filename = f'{base_filename}/graphs/{name}/{clustering}_{grammar_type}_{mu}_graphs.pkl'

        if not os.path.exists(graphs_filename):
            print('Graphs not found', graphs_filename)
            continue

        graph_list = pickle.load(open(graphs_filename, 'rb'))

        for i, g_test in enumerate(graph_list):
            g_test.name = f'{name}_{clustering}_{grammar_type}_{mu}_{i}'
            gcd, cvm_deg, cvm_pr, ld, dc0 = compare_two_graphs(g_true=g_true, g_test=g_test, true_deg=true_deg, true_page=true_pr)

            row = {'name': name, 'orig_n': g_true.order(), 'orig_m': g_true.size(), 'type': grammar_type, 'mu': mu, 'clustering': clustering,
                   'i': i+1, 'gen_n': g_test.order(), 'gen_m': g_test.size(), '#comps': nx.number_connected_components(g_test),
                   'gcd': gcd, 'deltacon0': dc0, 'lambda_dist': ld, 'cvm_deg': cvm_deg, 'cvm_pr': cvm_pr}

            with open(stats_filename, 'a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow(row)
    return


def main():
    np.seterr(all='ignore')
    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = 'karate'

    grammar_types = ('mu_random', 'mu_level', 'mu_dl', 'mu_level_dl', 'local_dl', 'global_dl')
    clustering_algs = ('cond', 'leiden', 'louvain', 'spectral', 'random')
    outdir = 'dumps'
    fieldnames = ('name', 'orig_n', 'orig_m', 'type', 'mu', 'clustering', 'i', 'gen_n', 'gen_m', '#comps',
                  'gcd', 'deltacon0', 'lambda_dist', 'cvm_deg', 'cvm_pr')

    make_dirs(outdir, name)  # make the directories if needed
    stats_path = f'{outdir}/gen_stats/{name}.csv'

    mode = 'w'
    if os.path.exists(stats_path):
        mode = 'a'

    if mode == 'w':
        with open(stats_path, mode) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    Parallel(n_jobs=15, verbose=1)(delayed(dump_graph_stats)(name=name, clustering=clustering, grammar_type=grammar_type)
                                  for grammar_type in grammar_types
                                   for clustering in clustering_algs)


if __name__ == '__main__':
    main()
