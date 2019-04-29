import networkx as nx
import os
import pickle
from time import time
import math
import logging

from src.VRG_new import VRG
from src.extract_new import MuExtractor
from src.Tree_new import create_tree, TreeNodeNew
import src.partitions as partitions
from src.LightMultiGraph import LightMultiGraph

def get_graph(filename='sample') -> LightMultiGraph:
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
    return g_new

def get_clustering(g, outdir, clustering) -> TreeNodeNew:
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

logging.basicConfig(level=logging.WARNING,
                    format="%(message)s")

def main():
    name = 'eucore'
    # name = 'eucore'
    outdir = 'output'
    clustering = 'leiden'
    type = 'mu_level_dl'
    mu = 3

    g = get_graph(name)
    root = get_clustering(g=g, outdir=f'{outdir}/trees/{name}', clustering=clustering)

    # TODO: watch out for label and b_deg - both are node attributes, but they are different levels

    grammar = VRG(clustering=clustering, type=type, name=name)
    extractor = MuExtractor(g=g, type='mu_random', mu=mu, grammar=grammar, root=root)
    extractor.generate_grammar()
    print(extractor.grammar)

if __name__ == '__main__':
    main()