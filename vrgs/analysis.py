"""
Script for analysis
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv

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


def compare_graphs(g_true, g_full, g_part, g_no, count=1):
    """
    Compares two graphs g1 and g2
    :param g1: graph 1
    :param g2: graph 2
    :return:
    """
    true_in = g_true.in_degree().values()
    true_out = g_true.out_degree().values()
    true_page = map(lambda x: round(x, 3), nx.pagerank_numpy(g_true).values())

    full_in = g_full.in_degree().values()
    full_out = g_full.out_degree().values()
    full_page = map(lambda x: round(x, 3), nx.pagerank_numpy(g_full).values())

    part_in = g_part.in_degree().values()
    part_out = g_part.out_degree().values()
    part_page = map(lambda x: round(x, 3), nx.pagerank_numpy(g_part).values())

    no_in = g_no.in_degree().values()
    no_out = g_no.out_degree().values()
    no_page = map(lambda x: round(x, 3), nx.pagerank_numpy(g_no).values())


    gcd_full = GCD(g_full, g_true)
    cdf_in_full = cdf_sum(full_in, true_in)
    cdf_out_full = cdf_sum(full_out, true_out)
    cdf_page_full = cdf_sum(full_page, true_page)

    gcd_part = GCD(g_part, g_true)
    cdf_in_part = cdf_sum(part_in, true_in)
    cdf_out_part = cdf_sum(part_out, true_out)
    cdf_page_part = cdf_sum(part_page, true_page)

    gcd_no = GCD(g_no, g_true)
    cdf_in_no = cdf_sum(no_in, true_in)
    cdf_out_no = cdf_sum(no_out, true_out)
    cdf_page_no = cdf_sum(no_page, true_page)

    with open('./stats.csv', 'a') as f:
        csvwriter = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([g_true.name, count, g_true.order(), g_true.size(), g_full.order(), g_full.size(),
                            g_part.order(), g_part.size(), g_no.order(), g_no.size(),
                            gcd_full, cdf_in_full, cdf_out_full, cdf_page_full,
                            gcd_part, cdf_in_part, cdf_out_part, cdf_page_part,
                            gcd_no, cdf_in_no, cdf_out_no, cdf_page_no])


def cdf_sum(data1, data2):
    data1, data2 = map(np.asarray, (data1, data2))
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    n1 = len(data1)
    n2 = len(data2)
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0 * n1)
    cdf2 = (np.searchsorted(data2, data_all, side='right')) / (1.0 * n2)
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