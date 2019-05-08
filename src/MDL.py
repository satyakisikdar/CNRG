import numpy as np
import math

def gamma_code(n):
    bits = math.log2(n)
    return 2 * bits + 1

def nbits(x):
    """
    Returns the number of bits to encode x in binary
    :param x: argument
    :return: number of bits required to encode x in binary
    """
    if x == 0:
        return 0
    return math.log2(x)


def graph_dl(g, l_u=2):
    """
     Get DL for graphs using Gamma coding
     :param g:  a multigraph
     :param l_u: number of unique labels in the graph - general graphs - 2, RHS graphs - 3 (2 types of nodes,
     1 type of edge)
     :return: Length in bits to represent graph g in binary
    """
    n = g.order()
    m = g.size()

    for node, data in g.nodes(data=True):
        if 'label' in data:
            l_u = 3
            break

    # encoding the nodes
    dl_v = nbits(n) + n * nbits(l_u)

    # encoding the edges
    dl_edges = 0
    for u, v in g.edges():
        k = g.number_of_edges(u, v)
        dl_edges += 2 * gamma_code(k + 1)  # 2 because the graph is undirected

    nnz = 2 * m  # the number of non-zero entries in the matrix
    dl_edges += (n ** 2 - nnz) * gamma_code(0 + 1)

    dl_e = nbits(m) + nbits(l_u) * dl_edges # added the l_u factor

    return dl_v + dl_e
