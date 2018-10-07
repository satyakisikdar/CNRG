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



def analyze_rules(vrg_rules, type='FULL'):
    """
    Analysis function for Rule objects
    :param vrg_rules:
    :return:
    """
    lvl_mdl = get_level_wise_mdl(vrg_rules)
    mean_mdl = {k: np.mean(v) for k, v in sorted(lvl_mdl.items())}

    xs, ys = split(mean_mdl)
    ys = np.cumsum(ys)
    plt.plot(xs, ys, marker='o', label=type, alpha=0.5)



def compare_two_graphs(g_true, g):
    """
    Compares two graphs
    :param g_true: actual graph
    :param g: generated graph
    :return:
    """
    true_deg = list(g_true.degree().values())
    true_page = list(map(lambda x: round(x, 3), nx.pagerank_numpy(g_true).values()))

    g_deg = list(g.degree().values())
    g_page = list(map(lambda x: round(x, 3), nx.pagerank_numpy(g).values()))

    gcd = GCD(g_true, g, mode='orca')
    cvm_deg = cvm_distance(true_deg, g_deg)
    cvm_page = cvm_distance(true_page, g_page)

    return gcd, cvm_deg, cvm_page


def compare_graphs(g_true, g_full, g_part, g_no,
                   graph_mdl, mdl_full, mdl_part, mdl_no,
                   k, count):
    """
    Compares two graphs g1 and g2
    Hop-plots, 90% diameter
    Average with 95% confidence intervals
    :param g1: graph 1
    :param g2: graph 2
    :return:
    """
    true_in = list(g_true.degree().values())
    true_page = list(map(lambda x: round(x, 3), nx.pagerank_numpy(g_true).values()))

    full_in = list(g_full.degree().values())
    full_page = list(map(lambda x: round(x, 3), nx.pagerank_numpy(g_full).values()))

    part_in = list(g_part.degree().values())
    part_page = list(map(lambda x: round(x, 3), nx.pagerank_numpy(g_part).values()))

    no_in = list(g_no.degree().values())
    no_page = list(map(lambda x: round(x, 3), nx.pagerank_numpy(g_no).values()))


    gcd_full = GCD(g_full, g_true)
    cdf_in_full = cdf_sum(full_in, true_in)
    cdf_page_full = cdf_sum(full_page, true_page)

    gcd_part = GCD(g_part, g_true)
    cdf_in_part = cdf_sum(part_in, true_in)
    cdf_page_part = cdf_sum(part_page, true_page)

    gcd_no = GCD(g_no, g_true)
    cdf_in_no = cdf_sum(no_in, true_in)
    cdf_page_no = cdf_sum(no_page, true_page)

    # add MDL
    with open('./stats_{}.csv'.format(k), 'a') as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if count == 1:  # for every network, insert a blank row
            csvwriter.writerow(['', '', '', '', '', '', '', '', '', '', '', '',
                                '', '', '', '', '', '', '', '', '', '', '', ''])

        csvwriter.writerow([g_true.name, count, k,
                            g_true.order(), g_true.size(),
                            g_full.order(), g_full.size(),
                            g_part.order(), g_part.size(),
                            g_no.order(), g_no.size(),
                            graph_mdl, mdl_full, mdl_part, mdl_no,
                            gcd_full, cdf_in_full, cdf_page_full,
                            gcd_part, cdf_in_part, cdf_page_part,
                            gcd_no, cdf_in_no, cdf_page_no])


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


def split(d):
    """
    Splits a dictionary of points (x, y) into two separate lists
    :param d: dictionary of points
    :return:
    """
    xs = []
    ys = []

    for k, v in sorted(d.items()):
        xs.append(k)
        ys.append(v)

    return xs, ys
