import networkx as nx
import platform
import subprocess
import pandas as pd
import scipy.stats
import math
import numpy as np

def GCD(h1, h2):
    df_g = external_rage(h1, 'Orig')
    df_h = external_rage(h2, 'Test')
    gcm_g = tijana_eval_compute_gcm(df_g)
    gcm_h = tijana_eval_compute_gcm(df_h)
    gcd = tijana_eval_compute_gcd(gcm_g, gcm_h)
    return round(gcd, 3)


def external_rage(G, netname):
    G = nx.Graph(G)
    giant_nodes = max(nx.connected_component_subgraphs(G), key=len)

    G = nx.subgraph(G, giant_nodes[0])
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

    popen = subprocess.Popen(args, stdout=subprocess.PIPE)

    popen.wait()
    output = popen.stdout.read()

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
    if len(gcm_h) != len(gcm_g):
        raise "Graphs must be same size"
    s = 0
    for i in range(0, len(gcm_g)):
        for j in range(i, len(gcm_h)):
            s += math.pow((gcm_g[i, j] - gcm_h[i, j]), 2)

    gcd = math.sqrt(s)
    return gcd