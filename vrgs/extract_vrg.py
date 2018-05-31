import random as r

import networkx as nx
from numpy import linalg
import numpy
import pandas as pd
from gensim.models import Word2Vec
from scipy.cluster.hierarchy import linkage, to_tree, cophenet
from scipy.spatial.distance import pdist

import node2vec #as node2vec


def get_graph(filename=None):
    if filename is not None:
        g = nx.read_edgelist(filename, nodetype=int, create_using=nx.MultiDiGraph())
    else:
        # g seems to be different than the dummy graph
        g = nx.MultiDiGraph()
        g.add_edge(1, 3)
        g.add_edge(2, 1)
        g.add_edge(2, 5)
        g.add_edge(3, 4)
        g.add_edge(4, 5)
        g.add_edge(4, 2)
        g.add_edge(4, 9)
        g.add_edge(5, 1)
        g.add_edge(5, 3)

        g.add_edge(6, 2)
        g.add_edge(6, 7)
        g.add_edge(6, 8)
        g.add_edge(6, 9)
        g.add_edge(7, 8)
        g.add_edge(9, 8)
        g.add_edge(9, 6)
    return g


def learn_embeddings(walks, filename='./tmp/temp.emb'):
    """
    Learn embeddings by optimizing the Skipgram objective using SGD.
    """
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=128, window=10, min_count=0, sg=1, workers=8, iter=1)
    model.wv.save_word2vec_format(filename)  ## TODO: keep in memory, dont write to file...


def get_embeddings(emb_filename='./tmp/temp.emb'):
    """
    g is undirected for the time being
    """
    df = pd.read_csv(emb_filename, skiprows=1, sep=' ', header=None)  ## maybe switch to Numpy read file functions
    return df.as_matrix()


def n2v_runner(g):
    nx_G = nx.Graph(g)
    nx.set_edge_attributes(nx_G, 'weight', 1)
    G = node2vec.Graph(nx_G, False, 1, 1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    learn_embeddings(walks)
    embeddings = get_embeddings()
    return embeddings


def get_dendrogram(embeddings, method='best', metric='euclidean'):
    """
    Generate dendrogram for graph.
    :param embeddings: node representations
    :param method: agglomoration measurement
    :param metric: distance metric
    :return: dendrogram of the graph nodes
    """
    methods = ('single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward')
    metrics = ('euclidean', 'cityblock', 'cosine', 'correlation', 'jaccard')
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


def find_boundary_edges(sg, g):
    """
    Collect all of the boundary edges (i.e., the edges
    that connect the subgraph to the original graph)

    :param sg: subgraph to remove from graph
    :param g: whole graph
    :return: boundary edges tuple of [0] indeges and [1] outedges. If undirected graph, then outedges will be empty
    """
    in_edges = list()
    out_edges = list()
    for n in sg:
        if g.is_directed():
            in_edges += g.in_edges(n)
            out_edges += g.out_edges(n)
        else:
            in_edges += g.edges(n)

    # remove internal edges from list of boundary_edges
    edge_set = set(sg.edges_iter())
    in_edges = [x for x in in_edges if x not in edge_set]
    out_edges = [x for x in out_edges if x not in edge_set]

    return in_edges, out_edges


def generalize_rhs(sg, internal_nodes):
    """
    Remove the original graph's labels from the RHS subgraph. Internal nodes are arabic characters, the boundary nodes
    are numerals.
    TODO - is this general enough? Can we make it easier to merge.

    :param sg: RHS subgraph
    :param internal_nodes: will be turned into arabic numerals
    :return: generalized subgraph.
    """
    nodes = {}
    internal_node_counter = 'a'  # maybe change to a1, a2, ..., ak?
    boundary_node_counter = 0
    rhs = nx.MultiDiGraph()

    for n in internal_nodes:
        rhs.add_node(internal_node_counter, sg.node[n])
        nodes[n] = internal_node_counter
        internal_node_counter = chr(ord(internal_node_counter) + 1)
    for n in [x for x in sg.nodes() if x not in internal_nodes]:
        rhs.add_node(boundary_node_counter, sg.node[n])
        nodes[n] = boundary_node_counter
        boundary_node_counter += 1
    for u, v, d in sg.edges(data=True):
        rhs.add_edge(nodes[u], nodes[v], attr_dict=d)
    return rhs


def extract_vrg(g, tree):
    """
    Extract a vertex replacement grammar (specifically an ed-NRC grammar) from a graph given a dendrogram tree

    :param g: graph to extract from
    :param tree: dendrogram with nodes at the bottom.
    :return: Vertex Replacement Grammar
    """
    vrg = list()
    if not isinstance(tree, list):
        # if we are at a leaf, then we need to backup one level
        return vrg
    for index, subtree in enumerate(tree):
        # build the grammar from a left-to-right bottom-up tree ordering (in order traversal)
        vrg += extract_vrg(g, subtree)
        if not isinstance(subtree, list):
            # if we are at a leaf, then we need to backup one level
            continue

        # subtree to be replaced
        print(subtree)

        sg = g.subgraph(subtree)
        print(sg.edges())
        boundary_edges = find_boundary_edges(sg, g)
        for direction in range(0, len(boundary_edges)):
            for u, v in boundary_edges[direction]:
                sg.add_edge(u, v, attr_dict={'b': True})

        lhs = (len(boundary_edges[0]), len(boundary_edges[1]))

        # next we contract the original graph
        [g.remove_node(n) for n in subtree]
        new_node = min(subtree)
        g.add_node(new_node, attr_dict={'label': lhs})

        # rewire new_node
        for direction in range(0, len(boundary_edges)):
            for u, v in boundary_edges[direction]:
                if u in subtree:
                    u = new_node
                if v in subtree:
                    v = new_node
                g.add_edge(u, v)

        rhs = generalize_rhs(sg, set(subtree))
        print(g.nodes(data=True))

        # replace subtree with new_node
        tree[index] = new_node
        vrg += [(lhs, rhs)]
    return vrg


def stochastic_vrg(vrg):
    """
    Create a new graph from the VRG at random
    :param vrg: Grammar used to generate
    :return: newly generated graph
    """

    node_counter = 1
    non_terminals = set()
    new_g = nx.MultiDiGraph()

    new_g.add_node(0, attr_dict={'label': (0, 0)})
    non_terminals.add(0)

    while len(non_terminals) > 0:
        # continue until no more non-terminal nodes

        # choose a non terminal node at random
        node_sample = r.sample(non_terminals, 1)[0]
        lhs = new_g.node[node_sample]['label']
        print('Selected node ' + str(node_sample) + ' with label ' + str(lhs))

        rhs = r.sample(vrg[lhs], 1)[0]

        singleton = nx.MultiDiGraph()
        singleton.add_node(node_sample)
        broken_edges = find_boundary_edges(singleton, new_g)
        assert (len(broken_edges[0]) == lhs[0] and len(broken_edges[1]) == lhs[1])

        new_g.remove_node(node_sample)
        non_terminals.remove(node_sample)

        nodes = {}

        for n, d in rhs.nodes(data=True):
            if isinstance(n, str):
                new_node = node_counter
                nodes[n] = new_node
                new_g.add_node(new_node, attr_dict=d)
                if 'label' in d:
                    non_terminals.add(new_node)
                node_counter += 1

        # randomly assign broken edges to boundary edges
        r.shuffle(broken_edges[0])
        r.shuffle(broken_edges[1])

        # wire the broken edge
        for u, v, d in rhs.edges(data=True):
            if 'b' in d:
                # boundry edge
                if isinstance(u, str):
                    # outedges
                    choice = r.sample(broken_edges[1], 1)[0]
                    new_g.add_edge(nodes[u], choice[1])
                    broken_edges[1].remove(choice)
                else:
                    # inedges
                    choice = r.sample(broken_edges[0], 1)[0]
                    new_g.add_edge(choice[0], nodes[v])
                    broken_edges[0].remove(choice)
            else:
                # internal edge
                new_g.add_edge(nodes[u], nodes[v])
    return new_g


def approx_min_conductance_partitioning(g, max_k=2):
    lvl = list()
    node_list = g.nodes()
    if len(node_list) <= max_k:
        return node_list
    print(node_list)
    fiedler_vector = nx.fiedler_vector(g.to_undirected())
    med = numpy.median(fiedler_vector)
    p1 = []
    p2 = []
    for idx, n in enumerate(fiedler_vector):
        if n > med:
            p1.append(node_list[idx])
        else:
            p2.append(node_list[idx])
    lvl.append(approx_min_conductance_partitioning(nx.subgraph(g, p1)))
    lvl.append(approx_min_conductance_partitioning(nx.subgraph(g, p2)))
    return [lvl]

def main():
    """
    Driver method for VRG
    :return:
    """
    g = get_graph('./tmp/dummy.txt')
    # embeddings = n2v_runner(g.copy())
    # tree = get_dendrogram(embeddings)
    # # print(tree)

    tree = approx_min_conductance_partitioning(g)
    print(tree)

    #tree = [[[[1,2], [[3,4], 5]], [[9,8], [6,7]]]]
    vrg = extract_vrg(g, tree)

    vrg_dict = {}
    # we need to turn the list into a dict for efficient access to the LHSs
    for lhs, rhs in vrg:
        if lhs not in vrg_dict:
            vrg_dict[lhs] = [rhs]
        else:
            vrg_dict[lhs].append(rhs)

    new_g = stochastic_vrg(vrg_dict)
    print('input graph degree distribution', nx.degree_histogram(get_graph()))
    print('output graph degree distribution', nx.degree_histogram(new_g))


if __name__ == '__main__':
    main()
