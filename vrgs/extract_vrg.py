import random as r

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.cluster.hierarchy import linkage, to_tree, cophenet
from scipy.spatial.distance import pdist
from time import time
import vrgs.node2vec as node2vec
from bitarray import bitarray
import math
from collections import Counter
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
# import node2vec


def get_graph(filename=None):
    if filename is not None:
        g = nx.read_edgelist(filename, nodetype=int, create_using=nx.MultiGraph())
        if nx.number_connected_components(g) > 0:
            g = max(nx.connected_component_subgraphs(g), key=len)
    else:
        g = nx.MultiGraph()

        g.add_edges_from([(1, 2), (1, 3), (1, 5),
                          (2, 4), (2, 5), (2, 7),
                          (3, 4), (3, 5),
                          (4, 5), (4, 9),
                          (6, 7), (6, 8), (6, 9),
                          (7, 8), (7, 9),
                          (8, 9)])

        # g.add_edge(1, 3)
        # g.add_edge(2, 1)
        # g.add_edge(2, 5)
        # g.add_edge(3, 4)
        # g.add_edge(4, 5)
        # g.add_edge(4, 2)
        # g.add_edge(4, 9)
        # g.add_edge(5, 1)
        # g.add_edge(5, 3)
        # g.add_edge(6, 2)
        # g.add_edge(6, 7)
        # g.add_edge(6, 8)
        # g.add_edge(6, 9)
        # g.add_edge(7, 8)
        # g.add_edge(9, 8)
        # g.add_edge(9, 6)
    return g


def learn_embeddings(walks, filename='./tmp/temp.emb'):
    """
    Learn embeddings by optimizing the Skipgram objective using SGD.
    """
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=128, window=10, min_count=0, sg=1, workers=8, iter=1)
    model.wv.save_word2vec_format(filename)  # TODO: keep in memory, dont write to file...


def get_embeddings(emb_filename='./tmp/temp.emb'):
    """
    g is undirected for the time being
    """
    df = pd.read_csv(emb_filename, skiprows=1, sep=' ', header=None)  # maybe switch to Numpy read file functions
    return df.as_matrix()


def n2v_runner(g):
    nx_g = nx.Graph(g)
    nx.set_edge_attributes(nx_g, 'weight', 1)
    g = node2vec.Graph(nx_g, False, 1, 1)
    g.preprocess_transition_probs()
    walks = g.simulate_walks(num_walks=10, walk_length=80)
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


def find_boundary_edges(g, nbunch):
    """
    Collect all of the boundary edges (i.e., the edges
    that connect the subgraph to the original graph)

    :param g: whole graph
    :param nbunch: set of nodes in the subgraph
    :return: boundary edges tuple of [0] indeges and [1] outedges. If undirected graph, then outedges will be empty
    """
    nbunch = set(nbunch)
    if g.is_directed():
        in_edges = [(u, v) for v in nbunch
                    for u in g.predecessors_iter(v)
                    if u not in nbunch]
        out_edges = [(u, v) for u in nbunch
                     for v in g.successors_iter(u)
                     if v not in nbunch]
    else:
        in_edges = [(u, v) for u in nbunch
                    for v in g.neighbors_iter(u)
                    if v not in nbunch]
        out_edges = []
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
    internal_node_counter = 'a'
    boundary_node_counter = 0
    # rhs = nx.MultiDiGraph()
    rhs = nx.MultiGraph()

    for n in internal_nodes:
        rhs.add_node(internal_node_counter, sg.node[n])
        nodes[n] = internal_node_counter
        internal_node_counter = chr(ord(internal_node_counter) + 1)

    for n in [x for x in sg.nodes_iter() if x not in internal_nodes]:
        rhs.add_node(boundary_node_counter, sg.node[n])
        nodes[n] = boundary_node_counter
        boundary_node_counter += 1

    for u, v, d in sg.edges_iter(data=True):
        rhs.add_edge(nodes[u], nodes[v], attr_dict=d)
    return rhs


clusters = {}  # stores the cluster members
original_graph = None   # to keep track of the edges covered
rule_coverage = []  # double check!


def deduplicate_edges(edges):
    """
    Takes an iterable of edges and makes sure there are no reverse edges
    :param edges: iterable of edges
    :return: uniq_edges: unique set of edges
    """
    uniq_edges = set()
    for u, v in edges:
        if (v, u) not in uniq_edges:
            uniq_edges.add((u, v))
    return uniq_edges


def extract_vrg(g, tree, lvl):
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
        vrg.extend(extract_vrg(g, subtree, lvl+1))
        if not isinstance(subtree, list):
            # if we are at a leaf, then we need to backup one level
            continue

        # subtree to be replaced
        # print(subtree, lvl)

        sg = g.subgraph(subtree)

        assert nx.number_connected_components(sg) == 1, "sg has > 1 components"

        nbunch = set()  # nbunch stores the set of original nodes in the graph
        for node in sg:
            if '_' in str(node):
                nbunch.update(clusters[node])
            else:
                nbunch.add(node)
        # print('st:', subtree, nbunch)

        edges_covered = set(original_graph.edges_iter(nbunch))  # this includes all the internal edges
        # print(sg.edges())
        boundary_edges = find_boundary_edges(g, subtree)

        for u, v in boundary_edges[0]:   # just iterates over the incoming edges since it's undirected
            if '_' in str(u) and '_' in str(v):   # u & v are clusters
                cut_edges = set(nx.edge_boundary(original_graph, clusters[u], clusters[v]))
                edges_covered.update(cut_edges)
                # print(u, v, clusters[u], clusters[v], cut_edges)
            elif '_' in str(u):  # u is a cluster
                cut_edges = set(nx.edge_boundary(original_graph, clusters[u], [v]))
                edges_covered.update(cut_edges)
                # print(u, v, clusters[u], cut_edges)
            elif '_' in str(v):  # v is a cluster
                cut_edges = set(nx.edge_boundary(original_graph, clusters[v], [u]))
                edges_covered.update(cut_edges)
                # print(u, v, clusters[v], cut_edges)
            else:  # both are nodes
                cut_edges = {(u, v)}
                edges_covered.update(cut_edges)
                # print(u, v, cut_edges)
        # print('edges covered', subtree, edges_covered)

        uniq_edges_covered = deduplicate_edges(edges_covered)

        for direction in range(len(boundary_edges)):
            for u, v in boundary_edges[direction]:
                sg.add_edge(u, v, attr_dict={'b': True})

        lhs = (len(boundary_edges[0]), len(boundary_edges[1]))

        # next we contract the original graph
        [g.remove_node(n) for n in subtree]

        new_node = min(subtree, key=lambda x: int(re.sub('_*', '', str(x))))
        new_node = '_{}'.format(new_node)  # each time you add an extra '_'

        # replace subtree with new_node
        tree[index] = new_node

        # print('before', clusters)
        clusters[new_node] = set()

        for node in subtree:
            if '_' in str(node):  # node is a cluster
                clusters[new_node].update(clusters[node])
                # del clusters[node]
            else:
                clusters[new_node].add(node)
        # print('after', clusters)
        g.add_node(new_node, attr_dict={'label': lhs})

        # rewire new_node
        subtree = set(subtree)
        for direction in range(len(boundary_edges)):
            for u, v in boundary_edges[direction]:
                if u in subtree:
                    u = new_node
                if v in subtree:
                    v = new_node
                g.add_edge(u, v)

        rhs = generalize_rhs(sg, set(subtree))

        assert nx.number_connected_components(rhs) == 1, "rhs has more than 1 component"

        rule_coverage.append((lhs[0], rhs, lvl, uniq_edges_covered, len(uniq_edges_covered) / original_graph.size()))

        # print(g.nodes(data=True))

        vrg.append((lhs, rhs))
    return vrg


def contract_grammar(vrg):
    """
    Contracts the right hand side of VRGs into Isolated boundary nodes if possible.
    :param vrg: list of VRG rules
    :return: reduced VRG
    """

    reduced_vrg = []

    for i, rule in enumerate(vrg):
        mapping = []  # mapping of isolated nodes for each rule
        lhs, rhs = rule  # LHS has a tuple (x, y): x is #incoming boundary edges, y is #outgoing boundary edges, RHS is a MultiDiGraph
        for node in rhs.nodes_iter():
            if isinstance(node, int) and rhs.degree(node) == 1:  # removing the isolated nodes
                mapping.append(node)

        # print(mapping)  # mapping now has the bounary nodes which can be contracted to a single node 'I'

        if len(mapping) == 0:
            reduced_vrg.append((lhs, rhs))
            continue

        new_rhs = rhs.copy()
        new_rhs.add_node('I')  # the new isolated node
        # rewire the edges to old isolated boundary nodes to the new isolated node
        for iso_node in mapping:
            new_rhs.remove_node(iso_node)
            if rhs.is_directed():
                for u in rhs.predecessors_iter(iso_node):
                    new_rhs.add_edge(u, 'I', attr_dict={'b': True})
                for v in rhs.successors_iter(iso_node):
                    new_rhs.add_edge('I', v, attr_dict={'b': True})
            else:
                for u in rhs.neighbors_iter(iso_node):
                    new_rhs.add_edge(u, 'I', attr_dict={'b': True})
        new_rhs_renumbered = nx.MultiGraph()
        # new_rhs_renumbered = nx.MultiDiGraph()
        mapper = {}
        cnt = 0
        for v, d in new_rhs.nodes_iter(data=True):
            if isinstance(v, int) and v not in mapper:
                mapper[v] = cnt
                cnt += 1
            else:
                mapper[v] = v
            new_rhs_renumbered.add_node(mapper[v], attr_dict=d)
        for u, v, d in new_rhs.edges_iter(data=True):
            new_rhs_renumbered.add_edge(mapper[u], mapper[v], attr_dict=d)

        g_old = nx.convert_node_labels_to_integers(new_rhs)
        g_new = nx.convert_node_labels_to_integers(new_rhs_renumbered)
        assert(nx.faster_could_be_isomorphic(g_old, g_new))

        reduced_vrg.append((lhs, new_rhs_renumbered))
    # print(reduced_vrg)

    return reduced_vrg


def deduplicate_vrg(vrg):
    """
    De-duplicates the grammar by merging the isomorphic RHSs
    :param vrg: dictionary of LHS -> list of RHSs
    :return: de-duplicated vrgs, with the frequency of the rules
    """
    dedup_vrg = {}
    iso_count = 0
    for rule in vrg:
        lhs, rhs = rule
        rhs = nx.freeze(rhs)
        if lhs not in dedup_vrg:  # first occurence, just put it in
            dedup_vrg[lhs] = [[rhs, 1]]  # first item of the rhs is the graph, the second is its frequency
        else:   # check for isomorphism
            g_new = nx.convert_node_labels_to_integers(rhs)
            existing_rhs = list(map(lambda x: x[0], dedup_vrg[lhs]))  # need to save the lhs to prevent looping over new ones
            isomorphic = False
            for i, g_old in enumerate(existing_rhs):
                g_o = nx.convert_node_labels_to_integers(g_old)
                if nx.is_isomorphic(g_new, g_o):
                    iso_count += 1
                    dedup_vrg[lhs][i][1] += 1
                    # print('iso:', lhs, i)
                    isomorphic = True
                    break
            if not isomorphic:  # new rule
                dedup_vrg[lhs].append([rhs, 1])
    print(iso_count, 'isomorphic rules')
    # print(dedup_vrg)

    return dedup_vrg


def stochastic_vrg(vrg):
    """
    Create a new graph from the VRG at random
    :param vrg: Grammar used to generate with the frequency of rules
    :return: newly generated graph
    """

    # normalize weights to probabilities
    # for lhs, rhs in vrg.items():
    #     sum_ = sum(map(lambda x: x[1], rhs))
    #     # vrg[lhs] = [x / sum_ for x in weight[lhs]]
    #     for i, rule in enumerate(rhs):
    #         rhs[i][1] = rule[1] / sum_
    print(vrg)
    return

    node_counter = 1
    non_terminals = set()
    new_g = nx.MultiGraph()
    # new_g = nx.MultiDiGraph()

    new_g.add_node(0, attr_dict={'label': (0, 0)})
    non_terminals.add(0)

    while len(non_terminals) > 0:
        # continue until no more non-terminal nodes

        # choose a non terminal node at random
        node_sample = r.sample(non_terminals, 1)[0]
        lhs = new_g.node[node_sample]['label']

        rhs_idx = int(np.random.choice(range(len(vrg[lhs])), size=1, p=weight[lhs]))
        rhs = vrg[lhs][rhs_idx]
        # print('Selected node ' + str(node_sample) + ' with label ' + str(lhs))
        # rhs = r.sample(vrg[lhs], 1)[0] ## Replace with funkier sampling
        max_v = -1
        for v in rhs.nodes_iter():
            if isinstance(v, int):
                max_v = max(v, max_v)
        max_v += 1
        for u, v in rhs.edges_iter():
            if u == 'I':
                rhs.remove_edge(u, v)
                rhs.add_edge(max_v, v, attr_dict={'b': True})
                max_v += 1
            elif v == 'I':
                rhs.remove_edge(u, v)
                rhs.add_edge(u, max_v, attr_dict={'b': True})
                max_v += 1

        if rhs.has_node('I'):
            assert(rhs.degree('I') == 0)
            rhs.remove_node('I')

        singleton = nx.MultiGraph()
        # singleton = nx.MultiDiGraph()
        singleton.add_node(node_sample)
        broken_edges = find_boundary_edges(singleton, new_g)
        assert (len(broken_edges[0]) == lhs[0] and len(broken_edges[1]) == lhs[1])

        new_g.remove_node(node_sample)
        non_terminals.remove(node_sample)

        nodes = {}

        for n, d in rhs.nodes_iter(data=True):
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
        for u, v, d in rhs.edges_iter(data=True):
            if 'b' in d:
                # boundary edge
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


def approx_min_conductance_partitioning(g, max_k):
    """
    Approximate minimum conductance partinioning. I'm using the median method as referenced here:
    http://www.ieor.berkeley.edu/~goldberg/pubs/krishnan-recsys-final2.pdf
    :param g: graph to recursively partition
    :param max_k:
    :return: a dendrogram
    """
    lvl = list()
    node_list = g.nodes()
    if len(node_list) <= max_k:
        assert(len(node_list) > 0)
        return node_list

    if g.is_directed():
        if not nx.is_weakly_connected(g):
            for p in nx.weakly_connected_component_subgraphs(g):
                lvl.append(approx_min_conductance_partitioning(p, max_k))
            assert (len(lvl) > 0)
            return lvl
    else:
        if not nx.is_connected(g):
            for p in nx.connected_component_subgraphs(g):
                lvl.append(approx_min_conductance_partitioning(p, max_k))
            assert  len(lvl) > 0
            return lvl

    assert nx.is_connected(g), "g is not connected in cond"

    fiedler_vector = nx.fiedler_vector(g, method='lobpcg')
    p1 = []
    p2 = []
    fiedler_dict = {}
    for idx, n in enumerate(fiedler_vector):
        fiedler_dict[idx] = n
    fiedler_vector = [(k, fiedler_dict[k]) for k in sorted(fiedler_dict,
                                                           key=fiedler_dict.get, reverse=True)]
    half_idx = len(fiedler_vector) // 2  # floor division

    for idx, _ in fiedler_vector:
        if half_idx > 0:
            p1.append(node_list[idx])
        else:
            p2.append(node_list[idx])
        half_idx -= 1  # decrement so halfway through it crosses 0 and puts into p2

    sg1 = g.subgraph(p1)
    sg2 = g.subgraph(p2)
    f1, f2 = False, False

    # Hack to check and fix non connected subgraphs
    if not nx.is_connected(sg1):
        f1 = True
        #maxsg1 = max(nx.connected_component_subgraphs(sg1), key = len)
        for sg in sorted(nx.connected_component_subgraphs(sg1), key=len, reverse=True)[1:]:
            p2.extend(sg.nodes())
            for n in sg.nodes():
                p1.remove(n)
    if not nx.is_connected(sg2):
        f2 = True
        #maxsg2 = max(nx.connected_component_subgraphs(sg2), key = len)
        for sg in sorted(nx.connected_component_subgraphs(sg2), key=len, reverse=True)[1:]:
            p1.extend(sg.nodes())
            for n in sg.nodes():
                p2.remove(n)

    if f1:
        sg1 = g.subgraph(p1)
    if f2:
        sg2 = g.subgraph(p2)

    # assert nx.is_connected(sg1) and nx.is_connected(sg2), "subgraphs not connected in cond"

    lvl.append(approx_min_conductance_partitioning(sg1, max_k))
    lvl.append(approx_min_conductance_partitioning(sg2, max_k))
    assert (len(lvl) > 0)
    return lvl


def gamma_code(n):
    binary_n = format(n, 'b')
    binary_offset = binary_n[1::]
    unary_length = bitarray(True for i in range(len(binary_offset))) + bitarray([False])
    return bitarray(unary_length) + bitarray(binary_offset)


def nCr(n, r):
    """
    Returns the value of n choose r.
    :param n: number of items
    :param r: number of items taken at a time
    :return: nCr
    """
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def nbits(x):
    """
    Returns the number of bits to encode x in binary
    :param x: argument
    :return: number of bits required to encode x in binary
    """
    if x == 0:
        return 0
    return math.ceil(math.log(x, 2))


def graph_mdl(g, l_u=2):
    """
    Get MDL for graphs
    Reference: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.407.6721&rep=rep1&type=pdf
    :param g: graph
    :param l_u: number of unique labels in the graph - general graphs - 2, RHS graphs - 4
    :return: Length in bits to represent graph G in binary
    """
    n = g.order()

    # encoding the nodes
    mdl_v = nbits(n) + n * nbits(l_u)

    # encoding the matrix - deal with a binary matrix and NOT actual number of edges - convert to DiGraph
    h = nx.DiGraph(g)

    b = max(h.out_degree().values())  # b is the max number of 1s in a row
    mdl_r = 0
    mdl_r += nbits(b + 1)

    for v in h.nodes_iter():
        k_i = h.out_degree(v)  # no of 1s in the v'th row
        # print('n = {}, k_i = {}'.format(n, k_i))
        mdl_r += nbits(b + 1) + nbits(nCr(n, k_i))

    # encoding the edges
    mdl_e = 0
    m = max(g.number_of_edges(u, v) for u, v in g.edges_iter())
    mdl_e += n ** 2 * nbits(m)

    for i, j in g.edges_iter():
        e_ij = g.number_of_edges(i, j)
        mdl_e += e_ij * (1 + nbits(l_u))

    return mdl_v + mdl_r + mdl_e


def graph_mdl_v2(g, l_u=2):
    """
    Get MDL for graphs using Gamma coding
    :param graph g
    :param number of unique labels in the graph - general graphs - 2, RHS graphs - 4
    :return: Length in bits to represent graph G in binary
    """
    n = g.order()

    # encoding the nodes
    mdl_v = nbits(n) + n * nbits(l_u)

    # encoding rows of matrix
    adj_mat = nx.to_numpy_matrix(g)

    mdl_r = 0
    for i in range(n):
        for j in range(n):
            a_ij = int(adj_mat[i, j])
            mdl_r += len(gamma_code(a_ij + 1))

    return mdl_v + mdl_r


def vrg_mdl(vrg):
    """
    MDL encoding for VRGs

    Each rule looks like
    LHS -> RHS_1 | RHS_2 | RHS_3 | ...
    represented in vrg as
    (x, y) -> [MultiDiGraph_1, MultiDiGraph_2, MultiDiGraph_3, ..]
    Must have a way to encode the edge - internal & boundary and node types
    :param vrg: vertex replacement grammar
    :return: MDL for vrg
    """
    mdl = 0

    num_rules = 0
    max_x = -1
    max_y = -1
    max_rhs_count = -1
    max_rhs_weight = -1
    all_rhs_graph_mdl = 0

    for lhs, rhs in vrg.items():
        x, y = lhs
        num_rules += len(rhs)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_rhs_count = max(max_rhs_count, len(rhs))
        max_rhs_weight = max(max_rhs_weight, max(map(lambda x: x[1], rhs)))
        for g, _ in rhs:
            all_rhs_graph_mdl += graph_mdl(g, l_u=4)   # 2 kinds of edges (boundary, non-boundary), 2 kinds of nodes (int/ext)
    # 1. number of rules
    mdl += nbits(num_rules)

    rule_mdl = 0  # mdl for one rule
    # 2. For each entry of the VRG dictionary,
    #    a. LHS has two numbers 'x' and 'y', upper bounded by max(x) and max(y)
    rule_mdl += nbits(max_x) + nbits(max_y)

    #    b. List of RHS graphs -
    #       i. count of RHSs upper bounded by max number of subgraphs for any LHS
    rule_mdl += nbits(max_rhs_count)  # for each rule, we need to store the count of RHSs

    #      ii. count of frequency of a given RHS - upper bounded by the max frequency of any LHS
    rule_mdl += nbits(max_rhs_count) * nbits(max_rhs_weight)  # for each RHS, store the frequency

    #     iii. The encoding of the RHS as a graph using graph_mdl + edge attributes for boundary edges (l_u = 4)
    # already done in the for loop

    mdl += all_rhs_graph_mdl + num_rules * rule_mdl  # adding it all up. rule_mdl gives MDL for one rule.

    # TODO: 3. The parse tree for lossless decomposition - dunno how tho!

    return mdl


def get_rhs_stats(vrg):
    rhs_counts = Counter()
    graph_set = set()

    for graphs in vrg.values():
        for g, count in graphs:
            # int_nodes_g = list(filter(lambda x: str(x) in string.ascii_lowercase, g.nodes_iter()))
            # g = nx.subgraph(g, int_nodes_g)
            g = nx.convert_node_labels_to_integers(g)
            g = nx.freeze(g)
            if len(graph_set) == 0:
                graph_set.add(g)
                rhs_counts[g] = 1
                continue

            for h in graph_set.copy():
                # int_nodes_h = list(filter(lambda x: str(x) in string.ascii_lowercase, h.nodes_iter()))
                # h = nx.subgraph(h, int_nodes_h)
                h = nx.convert_node_labels_to_integers(h)
                h = nx.freeze(h)
                if not nx.faster_could_be_isomorphic(g, h):
                    graph_set.add(g)
                    rhs_counts[g] = 1
                else:
                    print('iso!')
                    rhs_counts[h] += 1
                    break
    for g in graph_set:
        print('n = {} m = {}'.format(g.order(), g.size()))
    print(rhs_counts)


def get_freq_rhs(vrg):
    rule_counts = Counter()

    for _, rhs_list in vrg.items():
        for rhs, freq in rhs_list:
            if freq not in rule_counts:
                rule_counts[freq] = []
            rule_counts[freq].append(rhs)
    print(sorted(rule_counts.items(), reverse=True)[: 10])

    # plt.ylabel('#rules')
    # plt.xlabel('frequency')
    # # plt.xticks(range(min(rule_counts.keys()), math.ceil(max(rule_counts.keys())) + 1))
    # plt.bar(rule_counts.keys(), list(map(len, rule_counts.values())))
    # plt.show()

    print('Top 5 most frequent rules')
    for freq, rules in sorted(rule_counts.items(), reverse=True)[: 20]:
        if freq == 1:
            break
        for rule in rules:
            print('{}: {}'.format(freq, rule.edges()))

    # print('Top 5 infrequent rules')
    # for freq, rules in sorted(rule_counts.items())[: 5]:
    #     # if freq == 1:
    #     #     break
    #     for rule in rules:
    #         print('{}:  n = {}, m = {}'.format(freq, rule.order(), rule.size()))


def jaccard(set1, set2):
    """
    Returns the Jaccard coefficient between the two sets - intersection over union
    :param set1: the first set of tuples
    :param set2: the second set tuples
    :return: intersection over union
    """
    set1 = set(frozenset(item) for item in set1)  # tuples are ordered, so (x, y) and (y, x) are treated as different
    set2 = set(frozenset(item) for item in set2)
    return len(set1 & set2) / len(set1 | set2)


def rule_coverage_info():
    """
    Analyzes the rule coverage of vrgs
    :param:
    :return:
    """
    mdl_lvl = {}  # stores MDL for each level
    coverage_lvl = {} # stores the coverage for each level
    edges_lvl = {}

    max_lvl = max(rule_coverage, key=lambda x: x[2])[2]
    print('max_lvl', max_lvl)

    for item in rule_coverage:
        lhs, rhs, lvl, edges, f = item
        lvl = max_lvl - lvl

        if lvl == 8:
            print(rhs.edges())
            print()

        if lvl not in mdl_lvl:
            mdl_lvl[lvl] = []
        if lvl not in coverage_lvl:
            coverage_lvl[lvl] = []
        if lvl not in edges_lvl:
            edges_lvl[lvl] = []

        mdl = len(gamma_code(lhs + 1)) + graph_mdl_v2(rhs, 4)
        mdl_lvl[lvl].append(mdl)
        coverage_lvl[lvl].append(f)
        edges_lvl[lvl].append(edges)


    # mdl_list = [y / max_mdl for y in mdl_list]  # normalized mdl
    # print(mdl_lvl, coverage_lvl)

    ## Plot of level-wise rule coverage
    # avg_cov_lvl = dict((k, np.mean(v)) for k, v in coverage_lvl.items())
    #
    #
    # yerr = [np.std(v) for _, v in sorted(coverage_lvl.items())]
    # # print(yerr)
    # x, y = zip(*sorted(avg_cov_lvl.items()))
    #
    # cum_y = np.cumsum(y)

    # plt.title('Coverage v. height')
    # plt.xlabel('Height (leaves at 0)')
    # plt.ylabel('Fractional coverage')
    # plt.errorbar(x, y, yerr=yerr, marker='o', label='absolute')
    # # plt.plot(x, cum_y, marker='o', label='cumulative')
    # plt.legend(loc='best')
    # plt.show()

    # # Plot of level-wise rule MDL
    # avg_mdl_lvl = dict((k, np.mean(v)) for k, v in mdl_lvl.items())
    #
    # yerr = [np.std(v) for _, v in sorted(avg_mdl_lvl.items())]
    # print(yerr)
    # # x, y = zip(* sorted(avg_mdl_lvl))
    # x, y = zip(*sorted(avg_mdl_lvl.items()))
    # cum_y = np.cumsum(y)
    #
    # plt.title('MDL v. height')
    # plt.xlabel('Height (leaves at 0)')
    # plt.ylabel('Mean MDL')
    # plt.errorbar(x, y, yerr=yerr, marker='o', label='absolute')
    # plt.plot(x, cum_y, marker='o', label='cumulative')
    # plt.legend(loc='best')
    #
    #
    # for k, v in mdl_lvl.items():
    #     # plt.text(k, np.mean(v) + 50, '({}, {})'.format(len(v), np.sum(v)))
    #     print(k, len(v), np.sum(v), np.mean(v))
    # plt.show()



    # # pairwise intersection of coverage of all rules
    # cov_mat = np.zeros(shape=(len(rule_coverage), len(rule_coverage)))
    #
    # for i, item1 in enumerate(rule_coverage):
    #     _, _, _, edges1, _ = item1
    #     j = i
    #     for item2 in rule_coverage[i: ]:
    #         __, _, _, edges2, _ = item2
    #         cov_mat[i, j] = jaccard(edges1, edges2)
    #         cov_mat[j, i] = cov_mat[i, j]
    #         j += 1
    #
    # mask = np.zeros_like(cov_mat, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    # sns.heatmap(cov_mat, mask=mask, vmin=0, vmax=1, cmap='Reds')
    # # sns.clustermap(cov_mat, standard_scale=1)
    # plt.show()


def main():
    """
    Driver method for VRG
    :return:
    """
    global original_graph
    # g = get_graph()
    # g = get_graph('./tmp/karate.g')           # 34    78
    # g = get_graph('./tmp/lesmis.g')           # 77    254
    # g = get_graph('./tmp/football.g')         # 115   613
    # g = get_graph('./tmp/GrQc.g')             # 5,242 14,496
    # g = get_graph('./tmp/gnutella.g')         # 6,301 20,777
    # g = get_graph('./tmp/bitcoin_alpha.g')    # 3,783 24,186
    g = get_graph('./tmp/eucore.g')           # 1,005 25,571
    # g = get_graph('./tmp/bitcoin_otc.g')      # 5,881 35,592
    # g = get_graph('./tmp/wikivote.g')         # 7,115 103,689
    # g = get_graph('./tmp/Enron.g')            # 36,692 183,831
    # g = get_graph('./tmp/hepth.g')            # 27,770 352,807


    original_graph = g.copy()
    # g = nx.DiGraph(g)
    # embeddings = n2v_runner(g.copy())
    # tree = get_dendrogram(embeddings)
    # print(tree)

    # print('Graph MDL', graph_mdl(g), graph_mdl_v2(g))

    # old_g = g.copy()

    tree_time = time()

    k = 3
    print('k =', k)
    print('n = {}, m = {}'.format(g.order(), g.size()))
    tree = approx_min_conductance_partitioning(g, k)
    # print(tree)
    print('tree done in {} sec!'.format(time() - tree_time))

    # print(tree)
    vrg_time = time()
    vrg = extract_vrg(g, [tree], 1)
    print('VRG extracted in {} sec'.format(time() - vrg_time))
    print('#VRG rules: {}'.format(len(vrg)))
    # rule_coverage_info()
    #
    vrg = contract_grammar(vrg)
    vrg_dict = deduplicate_vrg(vrg)
    # # get_rhs_stats(vrg_dict)
    # get_freq_rhs(vrg_dict)

    print('VRG MDL', vrg_mdl(vrg_dict))
    get_rhs_stats(vrg_dict)
    return

    n_list = []
    m_list = []
    gen_time_list = []

    for n in range(10):
        gen_time = time()
        new_g = stochastic_vrg(vrg_dict, weight)
        n_list.append(new_g.order())
        m_list.append(new_g.size())
        gen_time_list.append(time() - gen_time)
        # print('old_g (n = {}, m = {})'.format(old_g.order(), old_g.size()))
        print('new_g (n = {}, m = {})'.format(new_g.order(), new_g.size()))
        # print('input graph degree distribution', nx.degree_histogram(old_g))
        # print('output graph degree distribution', nx.degree_histogram(new_g))

    print('stats: n = {}, m = {}, std(n) = {}, gen time = {} sec'.format(round(np.mean(n_list), 3),
                                                            round(np.mean(m_list), 3),
                                                            round(np.std(n_list), 3),
                                                            round(np.mean(gen_time_list), 3)))


if __name__ == '__main__':
    main()