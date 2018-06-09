import random as r

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy.cluster.hierarchy import linkage, to_tree, cophenet
from scipy.spatial.distance import pdist
from time import time
import vrgs.node2vec as node2vec
import math
# import node2vec


def get_graph(filename=None):
    if filename is not None:
        g = nx.read_edgelist(filename, nodetype=int, create_using=nx.MultiDiGraph())
        # if nx.number_weakly_connected_components(g) > 0:
        #     g = max(nx.weakly_connected_component_subgraphs(g), key=len)
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
    for n in sg.nodes_iter():
        if g.is_directed():
            in_edges.extend(g.in_edges(n))
            out_edges.extend(g.out_edges(n))
        else:
            in_edges.extend(g.edges(n))

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
    internal_node_counter = 'a'
    boundary_node_counter = 0
    rhs = nx.MultiDiGraph()

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
        vrg.extend(extract_vrg(g, subtree))
        if not isinstance(subtree, list):
            # if we are at a leaf, then we need to backup one level
            continue

        # subtree to be replaced
        # print(subtree)

        sg = g.subgraph(subtree)
        # print(sg.edges())
        boundary_edges = find_boundary_edges(sg, g)

        for direction in range(len(boundary_edges)):
            for u, v in boundary_edges[direction]:
                sg.add_edge(u, v, attr_dict={'b': True})

        lhs = (len(boundary_edges[0]), len(boundary_edges[1]))

        # next we contract the original graph
        [g.remove_node(n) for n in subtree]
        new_node = min(subtree)
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
        # print(g.nodes(data=True))

        # replace subtree with new_node
        tree[index] = new_node
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
            for u in rhs.predecessors_iter(iso_node):
                new_rhs.add_edge(u, 'I', attr_dict={'b': True})
            for v in rhs.successors_iter(iso_node):
                new_rhs.add_edge('I', v, attr_dict={'b': True})

        new_rhs_renumbered = nx.MultiDiGraph()
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
        assert(nx.is_isomorphic(g_old, g_new))

        reduced_vrg.append((lhs, new_rhs_renumbered))
    # print(reduced_vrg)

    return reduced_vrg


def deduplicate_vrg(vrg):
    """
    De-duplicates the grammar by merging the isomorphic RHSs
    :param vrg: dictionary of LHS -> list of RHSs
    :return: de-duplicated vrgs, weight of the rules
    """
    weight = {}
    dedup_vrg = {}
    iso_count = 0
    for rule in vrg:
        lhs, rhs = rule
        if lhs not in dedup_vrg:  # first occurence, just put it in
            dedup_vrg[lhs] = [rhs]
            weight[lhs] = [1]
        else:   # check for isomorphism
            g_new = nx.convert_node_labels_to_integers(rhs)
            existing_rhs = dedup_vrg[lhs].copy()  # need to save the lhs to prevent looping over new ones
            isomorphic = False
            for i, g_old in enumerate(existing_rhs):
                g_o = nx.convert_node_labels_to_integers(g_old)
                if nx.is_isomorphic(g_new, g_o):
                    iso_count += 1
                    weight[lhs][i] += 1
                    isomorphic = True
                    break
            if not isomorphic:  # new rule
                dedup_vrg[lhs].append(rhs)
                weight[lhs].append(1)
    print('#Isomorphic rules: ', iso_count)
    # print('weights:', weight)

    return dedup_vrg, weight


def stochastic_vrg(vrg, weight):
    """
    Create a new graph from the VRG at random
    :param vrg: Grammar used to generate
    :param weight: count of RHS occurences for a given LHS
    :return: newly generated graph
    """

    # normalize weights to probabilities
    for lhs, rhs in weight.items():
        sum_ = sum(weight[lhs])
        weight[lhs] = [x / sum_ for x in weight[lhs]]

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

        rhs_idx = int(np.random.choice(range(len(vrg[lhs])), size=1, p=weight[lhs]))
        rhs = vrg[lhs][rhs_idx]
        # print('Selected node ' + str(node_sample) + ' with label ' + str(lhs))
        # rhs = r.sample(vrg[lhs], 1)[0] ## Replace with funkier sampling
        max_v = -1
        for v in rhs.nodes_iter():
            if isinstance(v, int):
                max_v = max(v, max_v)
        max_v += 1
        for u, v in rhs.edges():
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

        singleton = nx.MultiDiGraph()
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
    # print(node_list)

    if not nx.is_weakly_connected(g):
        for p in nx.weakly_connected_component_subgraphs(g):
            lvl.append(approx_min_conductance_partitioning(p, max_k))
        assert (len(lvl) > 0)
        return lvl

    fiedler_vector = nx.fiedler_vector(g.to_undirected(), method='lanczos')
    p1 = []
    p2 = []
    fiedler_dict = {}
    for idx, n in enumerate(fiedler_vector):
        fiedler_dict[idx] = n
    fiedler_vector = [(k, fiedler_dict[k]) for k in sorted(fiedler_dict, key=fiedler_dict.get, reverse=True)]
    half_idx = len(fiedler_vector)//2  # floor division
    for idx, _ in fiedler_vector:
        if half_idx > 0:
            p1.append(node_list[idx])
        else:
            p2.append(node_list[idx])
        half_idx -= 1  # decrement so halfway through it crosses 0 and puts into p2

    lvl.append(approx_min_conductance_partitioning(nx.subgraph(g, p1), max_k))
    lvl.append(approx_min_conductance_partitioning(nx.subgraph(g, p2), max_k))
    assert (len(lvl) > 0)
    return [lvl]


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
    return math.ceil(math.log(x, 2))


def graph_mdl(g, l_u=2):
    """
    Get MDL for graphs
    Reference: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.407.6721&rep=rep1&type=pdf
    :param graph g
    :param number of unique labels in the graph - general graphs - 2, RHS graphs - 4
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


def vrg_mdl(vrg, weight):
    """
    MDL encoding for VRGs

    Each rule looks like
    LHS -> RHS_1 | RHS_2 | RHS_3 | ...
    represented in vrg as
    (x, y) -> [MultiDiGraph_1, MultiDiGraph_2, MultiDiGraph_3, ..]

    :param vrg: vertex replacement grammar
    :param weight: frequency of a rule
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
        max_rhs_weight = max(max_rhs_weight, max(weight[lhs]))
        for g in rhs:
            all_rhs_graph_mdl += graph_mdl(g, l_u=4)  # 2 kinds of edges (boundary, non-boundary), 2 kinds of nodes (int/ext)
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

def main():
    """
    Driver method for VRG
    :return:
    """
    # g = get_graph()
    # g = get_graph('./tmp/karate.g')
    # g = get_graph('./tmp/lesmis.g')
    g = get_graph('./tmp/football.g')
    # g = get_graph('./tmp/GrQc.g')
    # g = get_graph('./tmp/Enron.g')
    # g = get_graph('./tmp/Slashdot.g')
    # g = get_graph('./tmp/wikivote.g')
    # g = get_graph('./tmp/hepth.g')
    # g = nx.DiGraph(g)
    # embeddings = n2v_runner(g.copy())
    # tree = get_dendrogram(embeddings)
    # print(tree)
    g_mdl = graph_mdl(g)
    print('Graph MDL', g_mdl)

    old_g = g.copy()

    tree_time = time()
    k = 3
    print('k =', k)
    tree = approx_min_conductance_partitioning(g, k)
    print('n = {}, m = {}'.format(old_g.order(), old_g.size()))
    print('tree done in {} sec!'.format(time() - tree_time))

    # print(tree)
    vrg_time = time()
    vrg = extract_vrg(g, tree)
    print('VRG extracted in {} sec'.format(time() - vrg_time))
    print('#VRG rules: {}'.format(len(vrg)))

    vrg = contract_grammar(vrg)
    vrg_dict, weight = deduplicate_vrg(vrg)

    v_mdl = vrg_mdl(vrg_dict, weight)
    print('VRG MDL', v_mdl)
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
