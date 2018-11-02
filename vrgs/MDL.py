from bitarray import bitarray
import math
import networkx as nx
import sys
import numpy as np
import multiprocessing as mp

def gamma_code(n):
    bits = np.log2(n)
    return 2 * np.floor(bits) + 1


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


def graph_mdl_old(g, l_u=2):
    """
    Get MDL for graphs
    Reference: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.407.6721&rep=rep1&type=pdf
    :param g: graph
    :param l_u: number of unique labels in the graph - general graphs - 2, RHS graphs - 4
    :return: Length in bits to represent graph G in binary
    """
    print('entering graph mdl', file=sys.stderr)
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


def graph_mdl_seq(g, l_u=2):
    """
    Get MDL for graphs using Gamma coding
    :param g: graph
    :param l_u: number of unique labels in the graph - general graphs - 2, RHS graphs - 4 (2 types of nodes,
    2 types of edges)
    :return: Length in bits to represent graph g in binary
    """
    n = g.order()

    # encoding the nodes
    mdl_v = nbits(n) + n * nbits(l_u)

    # encoding rows of matrix
    mdl_r = 0

    # counting the upper triangle
    nnz = 2 * g.size()  # the number of non-zero entries in the matrix

    for u, v in g.edges_iter():
        k = g.number_of_edges(u, v)
        mdl_r += 2 * gamma_code(k + 1)

    mdl_r += (n ** 2 - nnz) * gamma_code(0 + 1)

    return mdl_v + mdl_r


mdl_r = 0
def edge_mdl_collector(mdl):
    global mdl_r
    mdl_r += mdl

def edge_fn(g, u, v):
    k = g.number_of_edges(u, v)
    return 2 * gamma_code(k + 1)

def graph_mdl(g, l_u=2):
    """
     Get MDL for graphs using Gamma coding
     :param g: graph
     :param l_u: number of unique labels in the graph - general graphs - 2, RHS graphs - 4 (2 types of nodes,
     2 types of edges)
     :return: Length in bits to represent graph g in binary
    """
    # global mdl_r
    n = g.order()

    # encoding the nodes
    mdl_v = nbits(n) + n * nbits(l_u)

    # encoding rows of matrix

    # counting the upper triangle
    nnz = 2 * g.size()  # the number of non-zero entries in the matrix

    mdl_r = 0
    for u, v in g.edges_iter():
        k = g.number_of_edges(u, v)
        mdl_r += 2 * gamma_code(k  + 1)
    # pool = mp.Pool(processes=20)
    #
    # for u, v in g.edges_iter():
    #     pool.apply_async(func=edge_fn, args=(g, u, v), callback=edge_mdl_collector)
    # pool.close()
    # pool.join()

    mdl_r += (n ** 2 - nnz) * len(gamma_code(0 + 1))
    # print('edge mdl', mdl_r, 'bits')

    return mdl_v + mdl_r


def vrg_mdl(vrg):
    """
    MDL encoding for VRGs

    Each rule looks like
    LHS -> RHS_1 | RHS_2 | RHS_3 | ...
    represented in vrg as
    (x, y) -> [MultiGraph_1, MultiGraph_2, MultiGraph_3, ..]
    Must have a way to encode the edge - internal & boundary and node types
    :param vrg: vertex replacement grammar
    :return: MDL for vrg
    """
    mdl = 0

    num_rules = 0
    max_x = -1
    max_rhs_count = -1
    max_rhs_weight = -1
    all_rhs_graph_mdl = 0

    for lhs, rhs in vrg.items():
        x = lhs
        num_rules += len(rhs)
        max_x = max(max_x, x)
        max_rhs_count = max(max_rhs_count, len(rhs))
        max_rhs_weight = max(max_rhs_weight, max(map(lambda x: x[1], rhs)))
        for g, _ in rhs:
            all_rhs_graph_mdl += graph_mdl(g, l_u=4)   # 2 kinds of edges (boundary, non-boundary), 2 kinds of nodes (int/ext)
    # 1. number of rules
    mdl += nbits(num_rules)

    rule_mdl = 0  # mdl for one rule
    # 2. For each entry of the VRG dictionary,
    #    a. LHS has one number 'x', upper bounded by max(x)
    rule_mdl += nbits(max_x)

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
