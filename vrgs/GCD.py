import networkx as nx
import os
import platform
import subprocess
import pandas as pd
import scipy.stats
import math
import numpy as np

def GCD(h1, h2, mode='rage'):
    if mode == 'rage':
        df_g = external_rage(h1, 'Orig')
        df_h = external_rage(h2, 'Test')
    else:
        df_g = external_orca(h1, 'orig')
        df_h = external_orca(h2, 'test')
    gcm_g = tijana_eval_compute_gcm(df_g)
    gcm_h = tijana_eval_compute_gcm(df_h)
    gcd = tijana_eval_compute_gcd(gcm_g, gcm_h)
    return round(gcd, 3)


def external_orca(g, gname):
    g = max(nx.connected_component_subgraphs(g), key=len)
    g = nx.convert_node_labels_to_integers(g, first_label=0)

    file_dir = './tmp'
    with open('{}/{}.in'.format(file_dir, gname), 'w') as f:
        f.write('{} {}\n'.format(g.order(), g.size()))
        for u, v in g.edges_iter():
            f.write('{} {}\n'.format(u, v))

    args = './orca', '4', '{}/{}.in'.format(file_dir, gname), '{}/{}.out'.format(file_dir, gname)

    popen = subprocess.Popen(args, stdout=subprocess.DEVNULL)
    popen.wait()

    df = pd.read_csv('{}/{}.out'.format(file_dir, gname), sep=' ', header=None)
    return df


def external_rage(G, netname):
    G = nx.Graph(G)
    G = max(nx.connected_component_subgraphs(G), key=len)

    tmp_file = "tmp_{}.txt".format(netname)
    with open(tmp_file, 'w') as tmp:
        for e in G.edges_iter():
            tmp.write(str(int(e[0]) + 1) + ' ' + str(int(e[1]) + 1) + '\n')

    if 'Windows' in platform.platform():
        args = ("./RAGE_windows.exe", tmp_file)
    elif 'Linux' in platform.platform():
        args = ('./RAGE_linux.dms', tmp_file)
    else:
        args = ("./RAGE.dms", tmp_file)

    popen = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    popen.wait()

    # Results are hardcoded in the exe
    df = pd.read_csv('./Results/UNDIR_RESULTS_tmp_{}.csv'.format(netname),
                     header=0, sep=',', index_col=0)
    df = df.drop('ASType', 1)
    return df


def tijana_eval_compute_gcm(G_df):
    l = len(G_df.columns)
    gcm = np.zeros((l, l))
    i = 0
    for column_G in G_df:
        j = 0
        for column_H in G_df:
            gcm[i, j] = scipy.stats.spearmanr(G_df[column_G].tolist(), G_df[column_H].tolist())[0]
            if scipy.isnan(gcm[i, j]):
                gcm[i, j] = 1.0
            j += 1
        i += 1
    return gcm


def tijana_eval_compute_gcd(gcm_g, gcm_h):
    assert len(gcm_h) == len(gcm_g), "Graphs must be same size"
    s = 0
    for i in range(len(gcm_g)):
        for j in range(i, len(gcm_h)):
            s += math.pow((gcm_g[i, j] - gcm_h[i, j]), 2)

    gcd = math.sqrt(s)
    return gcd