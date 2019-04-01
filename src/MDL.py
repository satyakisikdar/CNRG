from bitarray import bitarray
import math
import networkx as nx
import sys
import numpy as np
import multiprocessing as mp

def gamma_code(n):
    bits = np.log2(n)
    return 2 * bits + 1

def nbits(x):
    """
    Returns the number of bits to encode x in binary
    :param x: argument
    :return: number of bits required to encode x in binary
    """
    if x == 0:
        return 0
    return np.log2(x)


def graph_mdl(g, l_u=2):
    """
     Get MDL for graphs using Gamma coding
     :param g: graph
     :param l_u: number of unique labels in the graph - general graphs - 2, RHS graphs - 4 (2 types of nodes,
     2 types of edges)
     :return: Length in bits to represent graph g in binary
    """
    n = g.order()
    m = g.size()

    # encoding the nodes
    mdl_v = nbits(n) + n * nbits(l_u)

    # encoding the edges
    mdl_edges = 0
    for u, v in g.edges_iter():
        k = g.number_of_edges(u, v)
        mdl_edges += 2 * gamma_code(k + 1)  # 2 because the graph is undirected

    nnz = 2 * m  # the number of non-zero entries in the matrix
    mdl_edges += (n ** 2 - nnz) * gamma_code(0 + 1)

    mdl_e = nbits(m) + nbits(l_u) * mdl_edges # added the l_u factor

    return mdl_v + mdl_e
