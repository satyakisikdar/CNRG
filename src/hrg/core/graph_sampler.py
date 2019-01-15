__author__ = 'tweninge'
import networkx as nx
from random import choice
from collections import deque, Counter

def rwr_sample(G, c, n):
    for i in range(0,c):
        S = choice(G.nodes())

        T = nx.DiGraph()
        T.add_node(S)
        T.add_edges_from(bfs_edges(G,S,n))

        Gprime = nx.subgraph(G, T.nodes())

        yield Gprime

def rwr_sample_depth(G,S, n):

    T = nx.Graph()
    T.add_node(S)
    for a,e in nx.dfs_edges(G,S):
        T.add_edge(a,e)
        if T.number_of_nodes() >= n:
            break;

    Gprime = nx.subgraph(G, T.nodes())

    return Gprime

def ugander_sample(G):
    S = [0,0,0,0]
    a = choice(G.nodes())
    b = choice(G.nodes())
    c = choice(G.nodes())
    d = choice(G.nodes())

    Gprime = nx.subgraph(G, [a,b,c,d])

    return Gprime


def quad_graphs():
    qrect = nx.Graph()
    qrect.add_node(0)
    qrect.add_node(1)
    qrect.add_node(2)
    qrect.add_node(3)

    qrect.add_edge(0,1)
    qrect.add_edge(1,2)
    qrect.add_edge(2,3)
    qrect.add_edge(3,0)

    q4 = nx.Graph()
    q4.add_node(0)
    q4.add_node(1)
    q4.add_node(2)
    q4.add_node(3)

    q4.add_edge(0,1)
    q4.add_edge(1,2)
    q4.add_edge(2,0)
    q4.add_edge(2,3)

    return qrect, q4


def subgraphs_cnt(G, num_smpl):
    sub = Counter()
    sub['e2'] = 0
    sub['t2'] = 0
    sub['t3'] = 0
    sub['q3'] = 0
    sub['q4'] = 0
    sub['qrec'] = 0
    sub['q5'] = 0
    sub['q6'] = 0
    qrec,q4 = quad_graphs()
    for i in range(0,num_smpl):
        S = choice(G.nodes())
        #size 2
        T = rwr_sample_depth(G,S,2)
        sub['e2'] += 1

        T = rwr_sample_depth(G,S,3)
        if T.number_of_nodes() != 3:
            continue
        if T.number_of_edges() == 2:
            sub['t2'] += 1
        else:
            sub['t3'] += 1


        T = rwr_sample_depth(G,S,4)
        if T.number_of_nodes() != 4:
            continue
        if T.number_of_edges() == 3:
            sub['q3'] += 1
        elif T.number_of_edges() == 4 and nx.is_isomorphic(T, qrec):
            sub['qrec'] += 1
        elif T.number_of_edges() == 4 and nx.is_isomorphic(T, q4):
            sub['q4'] += 1
        elif T.number_of_edges() == 5:
            sub['q5'] += 1
        elif T.number_of_edges() == 6:
            sub['q6'] += 1
        else:
            print "error"

    return sub



def ugander_subgraphs_cnt(G, num_smpl):
    sub = Counter()
    sub['e0'] = 0
    sub['e1'] = 0
    sub['e2'] = 0
    sub['e2c'] = 0
    sub['tri'] = 0
    sub['p3'] = 0
    sub['star'] = 0
    sub['tritail'] = 0
    sub['square'] = 0
    sub['squarediag'] = 0
    sub['k4'] = 0
    for i in range(0,num_smpl):
        #size 2
        T = ugander_sample(G)
        #print T.edges()

        if T.number_of_edges() == 0:
            sub['e0'] += 1
        elif T.number_of_edges() == 1:
            sub['e1'] += 1
        elif T.number_of_edges() == 2:
            path = nx.Graph([(0,1), (1,2)])
            if len(max(nx.connected_component_subgraphs(T), key=len)) == 2:
                sub['e2'] += 1
            elif len(max(nx.connected_component_subgraphs(T), key=len)) == 3:
                sub['e2c'] += 1
            else:
                print "ERROR"
        elif T.number_of_edges() == 3:
            #triangle
            triangle = nx.Graph([(0,1), (1,2), (2,0)])
            #path
            path = nx.Graph([(0,1), (1,2), (2,3)])
            #star
            star = nx.Graph([(0,1), (0,2), (0,3)])
            if max(nx.connected_component_subgraphs(T), key=len).number_of_nodes() == 3:
                sub['tri'] += 1
            elif nx.is_isomorphic(T, path):
                sub['p3'] += 1
            elif nx.is_isomorphic(T, star):
                sub['star'] += 1
            else:
                print "ERROR"
        elif T.number_of_edges() == 4:
            square = nx.Graph([(0,1), (1,2), (2,3), (3,0)])
            triangletail = nx.Graph([(0,1), (1,2), (2,0), (2,3)])
            if nx.is_isomorphic(T, square):
                sub['square'] += 1
            elif nx.is_isomorphic(T, triangletail):
                sub['tritail'] += 1
            else:
                print "ERROR"
        elif T.number_of_edges() == 5:
            sub['squarediag'] += 1
        elif T.number_of_edges() == 6:
            sub['k3'] += 1
        else:
            print 'ERROR'

    return sub

def dfs_edges(G, source, n):
    nodes = [source]
    visited=set()
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start,iter(G[start]))]
        i=0
        while stack and i<n:
            parent,children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    i+=1
                    yield parent,child
                    visited.add(child)
                    stack.append((child,iter(G[child])))
            except StopIteration:
                stack.pop()

def bfs_edges(G, source, n):
    neighbors = G.neighbors_iter
    visited = set([source])
    queue = deque([(source, neighbors(source))])
    i=0
    while queue and i<n:
        parent, children = queue[0]
        try:
            child = next(children)
            if child not in visited:
                i += 1
                yield parent, child
                visited.add(child)
                queue.append((child, neighbors(child)))
        except StopIteration:
            queue.popleft()
