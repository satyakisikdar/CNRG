import itertools
import math
import os
import pickle
import random
import subprocess
from time import time

import networkx as nx
import networkx.algorithms.bipartite
import numpy as np
from networkx.generators.classic import empty_graph


def hrg_wrapper(g, n=5):
    print('Starting HRG...')
    nx.write_edgelist(g, './hrg/{}.g'.format(g.name), data=False)


    start_time = time()
    completed_process = subprocess.run('cd hrg; python2 exact_phrg.py --orig {}.g --trials {}'.format(g.name, n), shell=True)
    print('HRG ran in {} secs'.format(round(time() - start_time, 3)))

    if completed_process.returncode != 0:
        print('error in HRG')
        return None

    f = open('./hrg/Results/{}_hstars.pickle'.format(g.name), 'rb')
    graphs = pickle.load(f)
    assert len(graphs) == n, "HRG failed to generate {} graphs".format(n)

    return graphs


def bter_graphs(g, n=5):
    graphs = []
    for _ in range(n):
        graphs.append(bter_wrapper(g))
    return graphs


def bter_wrapper(g):
    # fix BTER to use the directory.. 
    print('Starting BTER...')
    np.savetxt('./bter/{}.mat'.format(g.name), nx.to_numpy_matrix(g), fmt='%d')

    matlab_code = []
    matlab_code.append("addpath('./bter');")
    matlab_code.append("G = dlmread('{}.mat');".format(g.name))
    matlab_code.append('G = sparse(G);')
    matlab_code.append("graphname = '{}';".format(g.name))
    matlab_code.append('')

    matlab_code.append('nnodes = size(G, 1);')
    matlab_code.append('nedges = nnz(G) / 2;')
    matlab_code.append(r"fprintf('nodes: %d edges: %d\n', nnodes, nedges);")
    matlab_code.append('')

    matlab_code.append('nd = accumarray(nonzeros(sum(G,2)),1);')
    matlab_code.append("maxdegree = find(nd>0,1,'last');")
    matlab_code.append(r"fprintf('Maximum degree: %d\n', maxdegree);")
    matlab_code.append('')

    matlab_code.append('[ccd,gcc] = ccperdeg(G);')
    matlab_code.append(r"fprintf('Global clustering coefficient: %.2f\n', gcc);")
    matlab_code.append('')

    matlab_code.append(r"fprintf('Running BTER...\n');")
    matlab_code.append('t1=tic;')
    matlab_code.append('[E1,E2] = bter(nd,ccd);')
    matlab_code.append('toc(t1);')
    matlab_code.append(r"fprintf('Number of edges created by BTER: %d\n', size(E1,1) + size(E2,1));")
    matlab_code.append('')

    matlab_code.append(r"fprintf('Turning edge list into adjacency matrix (including dedup)...\n');")
    matlab_code.append('t2=tic;')
    matlab_code.append('G_bter = bter_edges2graph(E1,E2);')
    matlab_code.append('toc(t2);')
    matlab_code.append(r"fprintf('Number of edges in dedup''d graph: %d\n', nnz(G)/2);")
    matlab_code.append('')

    matlab_code.append('G_bter = full(G_bter);')
    matlab_code.append(r"dlmwrite('{}_bter.mat', G_bter, ' ');".format(g.name))

    print('\n'.join(matlab_code), file=open('./bter/{}_code.m'.format(g.name), 'w'))
    
    if not os.path.isfile('./bter/{}_bter.mat'.format(g.name)):
        start_time = time()    
        completed_process = subprocess.run('cd bter; cat {}_code.m | matlab'.format(g.name), shell=True)
        print('BTER ran in {} secs'.format(round(time() - start_time, 3)))
    
        if completed_process.returncode != 0:
            print('error in matlab')
            return None

    bter_mat = np.loadtxt('./bter/{}_bter.mat'.format(g.name), dtype=int)

    g_bter = nx.from_numpy_matrix(bter_mat, create_using=nx.Graph())
    return g_bter


def chung_lu_graphs(g, n=5):
    graphs = []

    for _ in range(n):
        print('Starting Chung-Lu....')
        start_time = time()
        degree_seq = sorted(g.degree().values())
        g_chung_lu = nx.expected_degree_graph(degree_seq, selfloops=False)
        graphs.append(g_chung_lu)
        print('Chung-Lu ran in {} secs'.format(round(time() - start_time, 3)))

    return graphs


def kronecker_graph(k, G):
    I = nx.adjacency_matrix(G)

    if G.is_directed():
        return kronecker_random_graph(k, I, True)
    else:
        return kronecker_random_graph(k, I, False)


def kronecker_random_graph(k, P, seed=None, directed=True):
    """Return a random graph K_k[P] (Stochastic Kronecker graph).
    Parameters
    ----------
    P : square matrix of floats
        An n-by-n square "initiator" matrix of probabilities. May be a standard
        Python matrix or a NumPy matrix.  If the graph is undirected,
        must be symmetric.
    k : int
        The number of times P is Kronecker-powered, creating a stochastic
        adjacency matrix.  The generated graph has n^k nodes,
        where n is the dimension of P as noted above.
    seed : int, optional
        Seed for random number generator (default=None).
    directed : bool, optional
        If True, return a directed graph, else return an undirected one
        (default=True).
    Notes
    -----
    The stochastic Kronecker graph generation algorithm takes as input a
    square matrix of probabilities, computes the iterated Kronecker power of
    this matrix, and then uses the resulting stochastic adjacency matrix to
    generate a graph. This algorithm is O(V^2), where V=n^k.
    See Also
    --------
    kronecker2_random_graph
    Examples
    --------
    >>> k=4
    >>> P=[[0.8,0.3],[0.3,0.2]]
    >>> G=nx.kronecker_random_graph(k,P)
    >>> P=[[0.8,0.7],[0.3,0.2]]
    >>> G=nx.kronecker_random_graph(k,P,directed=True)

    References
    ----------
    .. [1] Jure Leskovec, Deepayan Chakrabarti, Jon Kleinberg, Christos Faloutsos,
           and Zoubin Ghahramani,
       "Kronecker graphs: an approach to modeling networks",
       The Journal of Machine Learning Research, 11, 985-1042, 3/1/2010.
    """
    dim = len(P)

    errorstring = ("The initiator matrix must be a nonempty" +
                   (", symmetric," if not directed else "") +
                   " square matrix of probabilities.")

    if dim == 0:
        raise nx.NetworkXError(errorstring)
    for i, arr in enumerate(P):
        if len(arr) != dim:
            raise nx.NetworkXError(errorstring)
        for j, p in enumerate(arr):
            if p < 0 or p > 1:
                raise nx.NetworkXError(errorstring)
            if not directed and P[i][j] != P[j][i]:
                raise nx.NetworkXError(errorstring)

    if k < 1:
        return empty_graph(1)

    n = dim ** k
    G = empty_graph(n)

    if directed:
        G = nx.DiGraph(G)

    G.add_nodes_from(range(n))
    G.name = "kronecker_random_graph(%s,%s)" % (n, P)

    if not seed is None:
        random.seed(seed)

    if G.is_directed():
        edges = itertools.product(range(n), range(n))
    else:
        edges = itertools.chain([(v, v) for v in range(n)], itertools.combinations(range(n), 2))

    for e in edges:
        row, col = e
        p = 1.0
        initPow = 1
        for i in range(k):
            rowVal = (row // initPow) % dim
            colVal = (col // initPow) % dim
            p = p * (P[rowVal][colVal])
            initPow = initPow * dim
        if random.random() < p:
            G.add_edge(*e)

    return G


def kronecker2_random_graph(k, P, seed=None, directed=True):
    """Return a sparse random graph K_k[P] (Stochastic Kronecker graph).
    Parameters
    ----------
    P : square matrix of floats
        An n-by-n square "initiator" matrix of probabilities. May be a standard
        Python matrix or a NumPy matrix.  If the graph is undirected,
        must be symmetric.
    k : int
        The number of times P is Kronecker-powered, creating a stochastic
        adjacency matrix.  The generated graph has n^k nodes,
        where n is the dimension of P as noted above.
    seed : int, optional
        Seed for random number generator (default=None).
    directed : bool, optional
        If True, return a directed graph, else return an undirected one
        (default=True).
    Notes
    -----
    The stochastic Kronecker graph generation algorithm takes as input a
    square matrix of probabilities, computes the iterated Kronecker power of
    this matrix, and then uses the resulting stochastic adjacency matrix to
    generate a graph.
    This "fast" algorithm runs in O(E) time. It thus works best when the expected
    number of edges in the graph is roughly O(V).
    The expected number of edges in the graph is given by d^k, where
    d=\sum_{i,j} P[i,j] is the sum of all the elements in P.
    See Also
    --------
    kronecker_random_graph
    Examples
    --------
    >>> k=4
    >>> P=[[0.8,0.3],[0.3,0.2]]
    >>> G=nx.kronecker2_random_graph(k,P)

    References
    ----------
    .. [1] Jure Leskovec, Deepayan Chakrabarti, Jon Kleinberg, Christos Faloutsos,
           and Zoubin Ghahramani,
       "Kronecker graphs: an approach to modeling networks",
       The Journal of Machine Learning Research, 11, 985-1042, 3/1/2010.
    """
    print('Starting Kronecker....')

    dim = len(P)

    errorstring = ("The initiator matrix must be a nonempty" +
                   (", symmetric," if not directed else "") +
                   " square matrix of probabilities.")

    if dim == 0:
        raise nx.NetworkXError(errorstring)
    for i, arr in enumerate(P):
        if len(arr) != dim:
            raise nx.NetworkXError(errorstring)
        for j, p in enumerate(arr):
            if p < 0 or p > 1:
                raise nx.NetworkXError(errorstring)
            if not directed and P[i][j] != P[j][i]:
                raise nx.NetworkXError(errorstring)

    if k < 1:
        return empty_graph(1)

    n = dim ** k
    G = empty_graph(n)
    G = nx.DiGraph(G)

    acc = 0.0
    partitions = []
    for i in range(dim):
        for j in range(dim):
            if P[i][j] != 0:
                acc = acc + P[i][j]
                partitions.append([acc, i, j])
    psum = acc

    G.add_nodes_from(range(n))
    G.name = "kronecker2_random_graph(%s,%s)" % (n, P)

    if not seed is None:
        random.seed(seed)

    expected_edges = math.floor(psum ** k)
    num_edges = 0
    while num_edges < expected_edges:
        multiplier = dim ** k
        x = y = 0
        for i in range(k):
            multiplier = multiplier // dim
            r = c = -1
            p = random.uniform(0, psum)
            for n in range(len(partitions)):
                if partitions[n][0] >= p:
                    r = partitions[n][1]
                    c = partitions[n][2]
                    break
            x = x + r * multiplier
            y = y + c * multiplier

        if not G.has_edge(x, y):
            G.add_edge(x, y)
            num_edges = num_edges + 1

    if not directed:
        G = G.to_undirected()

    return G


def subdue(g):
    print('Starting SUBDUE....')

    name = g.name
    g = nx.convert_node_labels_to_integers(g, first_label=1)
    g.name = name

    with open('./subdue/{}_subdue.g'.format(g.name), 'w') as f:
        for u in sorted(g.nodes_iter()):
            f.write('\nv {} v'.format(u))

        for u, v in g.edges_iter():
            f.write('\nu {} {} e'.format(u, v))

    start_time = time()

    completed_process = subprocess.run('cd subdue; ./subdue -undirected -nsubs 100000 {}_subdue.g'.format(g.name),
                                       shell=True, stdout=subprocess.PIPE)

    print('SUBDUE ran in {} secs'.format(round(time() - start_time, 3)))


    raw_st = completed_process.stdout.decode("utf-8")

    if completed_process.returncode != 0:
        return None

    structures = []


    lines = raw_st.split('\n')

    start_time = time()

    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith('('):  # start of a substructure
            sub_count = int(line.split(',')[-2].split()[-1])
            substr = nx.Graph()
            next_line = lines[i + 1]
            n = int(next_line[next_line.find('(') + 1: next_line.find('v')])  # number of nodes
            m = int(next_line[next_line.find(',') + 1: next_line.find('e')])  # number of edges

            for j in range(i + 2, i + 2 + m + n):
                if lines[j].strip().startswith('u'):  # it's an edge
                    u, v = map(int, lines[j].split()[1: 3])
                    substr.add_edge(u, v)
            i = j - 1
            structures.append((substr, sub_count))
        i += 1

    print('Parsing took {} secs'.format(round(time() - start_time, 3)))

    return structures


def vog(g):
    print('Starting VoG....')

    name = g.name
    g = nx.convert_node_labels_to_integers(g, first_label=1)

    with open('./vog/DATA/{}.g'.format(name), 'w') as f:
        for u, v in g.edges_iter():
            f.write('\n{} {} 1'.format(u, v))


    matlab_code = []
    matlab_code.append("addpath('DATA');")
    matlab_code.append("addpath('STRUCTURE_DISCOVERY');")


    matlab_code.append("input_file = './DATA/{}.g';".format(name))

    matlab_code.append("unweighted_graph = input_file;")
    matlab_code.append("output_model_greedy = 'DATA';")
    matlab_code.append("output_model_top10 = 'DATA';")
        
    matlab_code.append("orig = spconvert(load(input_file));")
    matlab_code.append("orig(max(size(orig)),max(size(orig))) = 0;")
    matlab_code.append("orig_sym = orig + orig';")
    matlab_code.append("[i,j,k] = find(orig_sym);")

    matlab_code.append("orig_sym(i(find(k==2)),j(find(k==2))) = 1;")
    matlab_code.append("orig_sym_nodiag = orig_sym - diag(diag(orig_sym));")

    matlab_code.append("disp('==== Running VoG for structure discovery ====')")
    matlab_code.append("global model;")
    matlab_code.append("model = struct('code', {}, 'edges', {}, 'nodes1', {}, 'nodes2', {}, 'benefit', {}, 'benefit_notEnc', {});")
    matlab_code.append("global model_idx;")
    matlab_code.append("model_idx = 0;")
    matlab_code.append("SlashBurnEncode( orig_sym_nodiag, 2, output_model_greedy, false, false, 3, unweighted_graph);")

    matlab_code.append("quit;")

    print('\n'.join(matlab_code), file=open('./vog/{}_vog_code.m'.format(name), 'w'))

    start_time = time()

    if not os.path.isfile('./vog/DATA/{}_ALL.model'.format(name)):
        completed_process = subprocess.run('cd vog; cat {}_vog_code.m | matlab'.format(name), shell=True)
        print('VoG ran in {} secs'.format(round(time() - start_time, 3)))


        if completed_process.returncode != 0:
            print('error in matlab')
            return None

    with open('./vog/DATA/{}_ALL.model'.format(name)) as f:
        raw_st = f.read()

    structures = []
    for line in raw_st.split('\n'):

        if line.startswith('fc'):  # a full clique "fc 1 2 3 4" => a clique on 4 nodes
            n = len(line.split()[1:])
            g_struct = nx.complete_graph(n)

        elif line.startswith('nc'):  # near clique "nc 5, 1 2 3 4" => a near clique over nodes 1-4 with 5 edges between them
            line = line.replace(',', '')
            m = int(line.split()[1])
            nodes = map(int, line.split()[2:])
            g_struct = g.subgraph(nodes)

        elif line.startswith('ch'):  # chain "ch 4 2 1 3" => 4-2-1-3
            n = len(line.split()[1:])
            g_struct = nx.path_graph(n)

        elif line.startswith('st'):  # star "st 1, 2 3 4" => star with 1 as hub with edges to 2, 3, 4
            line = line.replace(',', '')
            n_outer_nodes = len(line.split()[2:])
            g_struct = nx.star_graph(n_outer_nodes)

        elif line.startswith('bc'):  # bipartite core "bc 1 2 3, 4 5" complete bipartite graph between {1, 2, 3} and {4, 5}
            first_half, second_half = line.split(',')
            n1 = len(first_half.split()[1:])
            n2 = len(second_half.split())
            g_struct = networkx.bipartite.complete_bipartite_graph(n1, n2)

        elif line.startswith('nb'):  # near bipartite core "nb 1 2 3, 4 5"
            first_half, second_half = line.split(',')
            nodes_1 = set(map(int, first_half.split()[1:]))
            nodes_2 = set(map(int, second_half.split()))

            g_fb = nx.Graph()
            [g_fb.add_edge(u, v) for u in nodes_1 for v in nodes_2]  # making a complete biparite graph
            edges = set(g.edges_iter()) & set(g_fb.edges_iter())  # pick only those edges which actually exist in the graph

            g_struct = nx.Graph()
            g_struct.add_edges_from(edges)

        else:
            continue

        g_struct.name = line
        structures.append(g_struct)

    return structures
