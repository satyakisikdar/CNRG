"""
Script for analysis
"""
import networkx as nx
#import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import deque, Counter

from vrgs.GCD import GCD

def get_level_wise_mdl(vrg_rules):
    lvl_mdl = {}
    for rule in vrg_rules:
        rule.calculate_cost()
        if rule.level not in lvl_mdl:
            lvl_mdl[rule.level] = []
        lvl_mdl[rule.level].append(rule.cost)

    return lvl_mdl


def compare_two_graphs(g_true, g):
    """
    Compares two graphs
    :param g_true: actual graph
    :param g: generated graph
    :return:
    """
    true_deg = list(g_true.degree().values())
    true_page = list(map(lambda x: round(x, 3), nx.pagerank_scipy(g_true).values()))

    g_deg = list(g.degree().values())
    g_page = list(map(lambda x: round(x, 3), nx.pagerank_scipy(g).values()))

    gcd = GCD(g_true, g, 'orca')
    cvm_deg = cvm_distance(true_deg, g_deg)
    cvm_page = cvm_distance(true_page, g_page)

    return gcd, cvm_deg, cvm_page



def hop_plot(g):
    """
    Computes the hop-plot of graph g - number of nodes reachable in 'h' hops
    :param g: graph
    :return:
    """
    hop_counts = Counter()
    for u in g.nodes_iter():
        q = deque()
        nodes_covered = {u}
        q.append(u)
        hops = {u: 0}
        h = 0
        while len(q) != 0:
            u = q.popleft()
            h += 1
            for v in g.neighbors_iter(u):
                if v not in nodes_covered:
                    nodes_covered.add(v)
                    q.append(v)
                    hops[v] = hops[u] + 1
                    hop_counts[hops[v]] += 1

    return hop_counts



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
    return np.round(d, 3)
