"""
Contains the different partition methods
1. Conductance based partition
2. Node2vec hierarchical partition
"""

import vrgs.node2vec as n2v
import  networkx as nx
from scipy.cluster.hierarchy import linkage, to_tree, cophenet
from scipy.spatial.distance import pdist
import numpy as np
import scipy.sparse.linalg
from sklearn.cluster import KMeans
import random

def get_random_partition(g):
    nodes = g.nodes()
    random.shuffle(nodes)
    return random_partition(nodes)

def random_partition(nodes):
    tree = []
    if len(nodes) < 2:
        return nodes

    left = nodes[: len(nodes) // 2]
    right = nodes[len(nodes) // 2: ]

    tree.append(random_partition(left))
    tree.append(random_partition(right))

    return tree




def approx_min_conductance_partitioning(g, max_k=1):
    """
    Approximate minimum conductance partinioning. I'm using the median method as referenced here:
    http://www.ieor.berkeley.edu/~goldberg/pubs/krishnan-recsys-final2.pdf
    :param g: graph to recursively partition
    :param max_k: upper bound of number of nodes allowed in the leaves
    :return: a dendrogram
    """
    lvl = []
    node_list = g.nodes()
    if len(node_list) <= max_k:
        assert len(node_list) > 0
        return node_list

    if not nx.is_connected(g):
        for p in nx.connected_component_subgraphs(g):
            lvl.append(approx_min_conductance_partitioning(p, max_k))
        assert len(lvl) > 0
        return lvl

    assert nx.is_connected(g), "g is not connected in cond"

    fiedler_vector = nx.fiedler_vector(g, method='lanczos', tol=0)
    p1 = set()
    p2 = set()
    fiedler_dict = {}
    for idx, n in enumerate(fiedler_vector):
        fiedler_dict[idx] = n
    fiedler_vector = [(k, fiedler_dict[k]) for k in sorted(fiedler_dict,
                                                           key=fiedler_dict.get, reverse=True)]
    half_idx = len(fiedler_vector) // 2  # floor division

    for idx, _ in fiedler_vector:
        if half_idx > 0:
            p1.add(node_list[idx])
        else:
            p2.add(node_list[idx])
        half_idx -= 1  # decrement so halfway through it crosses 0 and puts into p2

    sg1 = g.subgraph(p1)
    sg2 = g.subgraph(p2)

    iter_count = 0
    while not (nx.is_connected(sg1) and nx.is_connected(sg2)):
        sg1 = g.subgraph(p1)
        sg2 = g.subgraph(p2)

        # Hack to check and fix non connected subgraphs
        if not nx.is_connected(sg1):
            for sg in sorted(nx.connected_component_subgraphs(sg1), key=len, reverse=True)[1: ]:
                p2.update(sg.nodes_iter())
                for n in sg.nodes_iter():
                    p1.remove(n)

            sg2 = g.subgraph(p2)  # updating sg2 since p2 has changed

        if not nx.is_connected(sg2):
            for sg in sorted(nx.connected_component_subgraphs(sg2), key=len, reverse=True)[1: ]:
                p1.update(sg.nodes_iter())
                for n in sg.nodes_iter():
                    p2.remove(n)

        iter_count += 1

    if iter_count > 2:
        print('it took {} iterations to stabilize'.format(iter_count))


    assert nx.is_connected(sg1) and nx.is_connected(sg2), "subgraphs are not connected in cond"

    lvl.append(approx_min_conductance_partitioning(sg1, max_k))
    lvl.append(approx_min_conductance_partitioning(sg2, max_k))

    assert (len(lvl) > 0)
    return lvl


def spectral_kmeans(g, k):
    """
    k-way ncut spectral clustering Ng et al. 2002 KNSC1
    :param g: graph g
    :param k: number of clusters
    :return:
    """
    tree = []

    if g.order() <= k:   # not more than k nodes, return the list of nodes
        return g.nodes()

    if k == 2:  # if k is two, use approx min partitioning
        return [approx_min_conductance_partitioning(g, 1)]

    if not nx.is_connected(g):
        for p in nx.connected_component_subgraphs(g):
            if p.order() > k + 1:   # if p has more than k + 1 nodes, use spectral k-means
                tree.append(spectral_kmeans(p, k))
            else:   # try spectral k-means with a lesser k
                tree.append(spectral_kmeans(p, k - 1))
        assert len(tree) > 0
        return tree

    if k >= g.order() - 2:
        return spectral_kmeans(g, k - 1)

    assert nx.is_connected(g), "g is not connected in cond"

    L = nx.laplacian_matrix(g)

    assert k < g.order() - 2, "k is too high"

    _, eigenvecs = scipy.sparse.linalg.eigs(L.asfptype(), k=k + 1, which='SM')  # compute the first k+1 eigenvectors
    eigenvecs = eigenvecs[:, 1:]  # discard the first trivial eigenvector

    U = np.apply_along_axis(lambda x: x / np.linalg.norm(x), axis=1, arr=eigenvecs)  # normalize each row by its norm

    kmeans = KMeans(n_clusters=k, random_state=0).fit(U)

    cluster_labels = kmeans.labels_
    clusters = [[] for _ in range(max(cluster_labels) + 1)]

    for u, clu_u in zip(g.nodes_iter(), cluster_labels):
        clusters[clu_u].append(u)

    for cluster in clusters:
        sg = g.subgraph(cluster)
        assert nx.is_connected(sg), "subgraph not connected"
        if len(cluster) > k + 1:
            tree.append(spectral_kmeans(sg, k))
        else:
            tree.append(spectral_kmeans(sg, k - 1))

    return tree


def get_dendrogram(embeddings, method='best', metric='euclidean'):
    """
    Generate dendrogram for graph.
    :param embeddings: node representations
    :param method: agglomoration measurement
    :param metric: distance metric
    :return: dendrogram of the graph nodes
    """
    methods = ('single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward')
    # metrics = ('euclidean', 'cityblock', 'cosine', 'correlation', 'jaccard')
    # centroid, median, ward only work with Euclidean

    if method not in methods and method != 'best':
        print('Invalid method {}. Choosing an alternative model instead.')
        method = 'best'

    if method == 'best':
        best_method = None
        best_score = 0

        for method in methods:
            z = linkage(embeddings[:, 1:], method)
            c, _ = cophenet(z, pdist(embeddings[:, 1:], metric))
            print(method, metric, c)
            if c > best_score:
                best_score = c
                best_method = method
        print('Using "{}, {}" for clustering'.format(best_method, metric))
        z = linkage(embeddings[:, 1:], best_method)
    else:
        z = linkage(embeddings[:, 1:], method)

    root = to_tree(z)
    labels = list(map(int, embeddings[:, 0]))

    def print_tree(node):
        if node.is_leaf():  # single leaf
            return [labels[node.id]]

        if node.left.is_leaf() and node.right.is_leaf():  # combine two leaves into one
            return [labels[node.left.id], labels[node.right.id]]

        left_list = print_tree(node.left)
        right_list = print_tree(node.right)
        return [left_list, right_list]

    return [print_tree(root)]


def n2v_partition(g):
    """
    Partitions the graph using hierarchical clustering on node2vec embeddings
    :param g: graph
    :return: tree of partitions
    """
    nx_g = nx.Graph(g)
    nx.set_edge_attributes(nx_g, 'weight', 1)
    g = n2v.Graph(nx_g, False, 1, 1)
    g.preprocess_transition_probs()
    walks = g.simulate_walks(num_walks=10, walk_length=80)
    n2v.learn_embeddings(walks)
    embeddings = n2v.get_embeddings()
    return get_dendrogram(embeddings)

