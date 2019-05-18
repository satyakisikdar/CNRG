import platform
import subprocess

import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial
import scipy.stats

np.seterr(all='ignore')


def GCD(h1, h2, mode='orca'):
    if mode == 'rage':
        df_g = external_rage(h1, '{}_o'.format(h1.name))
        df_h = external_rage(h2, '{}_t'.format(h2.name))
    else:
        df_g = external_orca(h1, '{}_o'.format(h1.name))
        df_h = external_orca(h2, '{}_t'.format(h2.name))

    gcm_g = tijana_eval_compute_gcm(df_g)
    gcm_h = tijana_eval_compute_gcm(df_h)

    gcd = tijana_eval_compute_gcd(gcm_g, gcm_h)
    return round(gcd, 3)


def external_orca(g: nx.Graph, gname: str):
    g = nx.Graph(g)  # convert it into a simple graph
    g = max(nx.connected_component_subgraphs(g), key=len)
    selfloops = g.selfloop_edges()
    g.remove_edges_from(selfloops)   # removing self-loop edges

    g = nx.convert_node_labels_to_integers(g, first_label=0)

    file_dir = 'src/tmp'
    with open(f'./{file_dir}/{gname}.in', 'w') as f:
        f.write(f'{g.order()} {g.size()}\n')
        for u, v in g.edges():
            f.write(f'{u} {v}\n')

    args = ['', '4', f'./{file_dir}/{gname}.in', f'./{file_dir}/{gname}.out']

    if 'Windows' in platform.platform():
        args[0] = './src/orca.exe'
    elif 'Linux' in platform.platform():
        args[0] = './src/orca_linux'
    else:
        args[0] = './src/orca_mac'

    process = subprocess.run(' '.join(args), shell=True, stdout=subprocess.DEVNULL)
    if process.returncode != 0:
        print('Error in ORCA')

    df = pd.read_csv(f'./{file_dir}/{gname}.out', sep=' ', header=None)
    return df


def external_rage(G, netname):
    G = nx.Graph(G)
    G = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.convert_node_labels_to_integers(G, first_label=1)

    tmp_file = "tmp_{}.txt".format(netname)
    nx.write_edgelist(G, tmp_file, data=False)

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


def spearmanr(x, y):
    """
    Spearman correlation - takes care of the nan situation
    :param x:
    :param y:
    :return:
    """
    score = scipy.stats.spearmanr(x, y)[0]
    if np.isnan(score):
        score = 1
    return score


def tijana_eval_compute_gcm(G_df):
    l = G_df.shape[1]  # no of graphlets: #cols in G_df

    M = G_df.values  # matrix of nodes & graphlet counts
    M = np.transpose(M)  # transpose to make it graphlet counts & nodes
    gcm = scipy.spatial.distance.squareform(   # squareform converts the sparse matrix to dense matrix
        scipy.spatial.distance.pdist(M,   # compute the pairwise distances in M
                                     spearmanr))  # using spearman's correlation
    gcm = gcm + np.eye(l, l)   # make the diagonals 1 (dunno why, but it did that in the original code)
    return gcm


def tijana_eval_compute_gcd(gcm_g, gcm_h):
    assert len(gcm_h) == len(gcm_g), "Correlation matrices must be of the same size"

    gcd = np.sqrt(   # sqrt
        np.sum(  # of the sum of elements
            (np.triu(gcm_g) - np.triu(gcm_h)) ** 2   # of the squared difference of the upper triangle values
        ))
    if np.isnan(gcd):
        print('GCD is nan')
    return round(gcd, 3)