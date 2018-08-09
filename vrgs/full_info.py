"""
Full info extraction and generation

Uses explicit boundary information containing node level info on boundary nodes and edges
"""

import networkx as nx
import re
import vrgs.globals as globals
from vrgs.Rule import Rule


def find_boundary_edges(g, nbunch):
    """
    Collect all of the boundary edges (i.e., the edges
    that connect the subgraph to the original graph)

    :param g: whole graph
    :param nbunch: set of nodes in the subgraph
    :return: boundary edges
    """
    nbunch = set(nbunch)
    return [(u, v)
            for u in nbunch
            for v in g.neighbors_iter(u)
            if v not in nbunch]


def deduplicate_edges(edges):
    """
    Takes an iterable of edges and makes sure there are no reverse edges
    :param edges: iterable of edges
    :return: uniq_edges: unique set of edges
    """
    uniq_edges = set()
    for u, v in edges:
        if (v, u) not in uniq_edges:
            uniq_edges.add((u, v))
    return uniq_edges



def extract_vrg(g, tree, lvl):
    """
    Extract a vertex replacement grammar (specifically an ed-NRC grammar) from a graph given a dendrogram tree


    :param g: graph to extract from
    :param tree: dendrogram with nodes at the bottom.
    :return: List of Rule objects
    """
    rule_list = list()
    if not isinstance(tree, list):
        # if we are at a leaf, then we need to backup one level
        return rule_list
    for index, subtree in enumerate(tree):
        # build the grammar from a left-to-right bottom-up tree ordering (in order traversal)
        rule_list.extend(extract_vrg(g, subtree, lvl+1))
        if not isinstance(subtree, list):
            # if we are at a leaf, then we need to backup one level
            continue

        # subtree to be replaced
        # print(subtree, lvl)

        sg = g.subgraph(subtree)

        nbunch = set()  # nbunch stores the set of original nodes in the graph
        for node in sg:
            if '_' in str(node):
                nbunch.update(globals.clusters[node])
            else:
                nbunch.add(node)
        # print('st:', subtree, nbunch)

        edges_covered = set(globals.original_graph.edges_iter(nbunch))  # this includes all the internal edges
        # print(sg.edges())
        boundary_edges = find_boundary_edges(g, subtree)

        for u, v in boundary_edges:   # just iterates over the incoming edges since it's undirected
            if '_' in str(u) and '_' in str(v):   # u & v are globals.clusters
                cut_edges = set(nx.edge_boundary(globals.original_graph, globals.clusters[u], globals.clusters[v]))
                edges_covered.update(cut_edges)
                # print(u, v, globals.clusters[u], globals.clusters[v], cut_edges)
            elif '_' in str(u):  # u is a cluster
                cut_edges = set(nx.edge_boundary(globals.original_graph, globals.clusters[u], [v]))
                edges_covered.update(cut_edges)
                # print(u, v, globals.clusters[u], cut_edges)
            elif '_' in str(v):  # v is a cluster
                cut_edges = set(nx.edge_boundary(globals.original_graph, globals.clusters[v], [u]))
                edges_covered.update(cut_edges)
                # print(u, v, globals.clusters[v], cut_edges)
            else:  # both are nodes
                cut_edges = {(u, v)}
                edges_covered.update(cut_edges)
                # print(u, v, cut_edges)
        # print('edges covered', subtree, edges_covered)

        for u, v in boundary_edges:
            sg.add_edge(u, v, attr_dict={'b': True})

        rule = Rule()
        rule.lhs = len(boundary_edges)
        rule.rhs = sg
        rule.internal_nodes = subtree
        rule.level = lvl
        rule.edges_covered = deduplicate_edges(edges_covered)
        rule.generalize_rhs()
        rule.contract_rhs()

        # next we contract the original graph
        [g.remove_node(n) for n in subtree]

        new_node = min(subtree, key=lambda x: int(re.sub('_*', '', str(x))))
        tree[index] = new_node
        g.add_node(new_node, attr_dict={'label': rule.lhs})


        # rewire new_node
        subtree = set(subtree)

        for u, v in boundary_edges:
            if u in subtree:
                u = new_node
            if v in subtree:
                v = new_node
            g.add_edge(u, v)


        new_node = '_{}'.format(new_node)  # each time you add an extra '_'

        # replace subtree with new_node



        # print('before', globals.clusters)
        globals.clusters[new_node] = set()

        for node in subtree:
            if '_' in str(node):  # node is a cluster
                globals.clusters[new_node].update(globals.clusters[node])
            else:
                globals.clusters[new_node].add(node)


        assert nx.is_connected(rule.rhs) == 1, "rhs is not connected"

        # print(g.nodes(data=True))

        rule_list.append(rule)
    return rule_list


def generate_graph(vrg):
    """
    Create a new graph from the VRG at random
    :param vrg: Grammar used to generate with the frequency of rules
    :return: newly generated graph
    """

    # normalize weights to probabilities
    # for lhs, rhs in vrg.items():
    #     sum_ = sum(map(lambda x: x[1], rhs))
    #     # vrg[lhs] = [x / sum_ for x in weight[lhs]]
    #     for i, rule in enumerate(rhs):
    #         rhs[i][1] = rule[1] / sum_
    print(vrg)
    return

    node_counter = 1
    non_terminals = set()
    new_g = nx.MultiGraph()
    # new_g = nx.MultiDiGraph()

    new_g.add_node(0, attr_dict={'label': (0, 0)})
    non_terminals.add(0)

    while len(non_terminals) > 0:
        # continue until no more non-terminal nodes

        # choose a non terminal node at random
        node_sample = r.sample(non_terminals, 1)[0]
        lhs = new_g.node[node_sample]['label']

        rhs_idx = int(np.random.choice(range(len(vrg[lhs])), size=1, p=weight[lhs]))
        rhs = vrg[lhs][rhs_idx]
        # print('Selected node ' + str(node_sample) + ' with label ' + str(lhs))
        # rhs = r.sample(vrg[lhs], 1)[0] ## Replace with funkier sampling
        max_v = -1
        for v in rhs.nodes_iter():
            if isinstance(v, int):
                max_v = max(v, max_v)
        max_v += 1
        for u, v in rhs.edges_iter():
            if u == 'I':
                rhs.remove_edge(u, v)
                rhs.add_edge(max_v, v, attr_dict={'b': True})
                max_v += 1
            elif v == 'I':
                rhs.remove_edge(u, v)
                rhs.add_edge(u, max_v, attr_dict={'b': True})
                max_v += 1

        if rhs.has_node('I'):
            assert(rhs.degree('I') == 0)
            rhs.remove_node('I')

        singleton = nx.MultiGraph()
        # singleton = nx.MultiDiGraph()
        singleton.add_node(node_sample)
        broken_edges = find_boundary_edges(singleton, new_g)
        assert (len(broken_edges[0]) == lhs[0] and len(broken_edges[1]) == lhs[1])

        new_g.remove_node(node_sample)
        non_terminals.remove(node_sample)

        nodes = {}

        for n, d in rhs.nodes_iter(data=True):
            if isinstance(n, str):
                new_node = node_counter
                nodes[n] = new_node
                new_g.add_node(new_node, attr_dict=d)
                if 'label' in d:
                    non_terminals.add(new_node)
                node_counter += 1

        # randomly assign broken edges to boundary edges
        r.shuffle(broken_edges[0])
        r.shuffle(broken_edges[1])

        # wire the broken edge
        for u, v, d in rhs.edges_iter(data=True):
            if 'b' in d:
                # boundary edge
                if isinstance(u, str):
                    # outedges
                    choice = r.sample(broken_edges[1], 1)[0]
                    new_g.add_edge(nodes[u], choice[1])
                    broken_edges[1].remove(choice)
                else:
                    # inedges
                    choice = r.sample(broken_edges[0], 1)[0]
                    new_g.add_edge(choice[0], nodes[v])
                    broken_edges[0].remove(choice)
            else:
                # internal edge
                new_g.add_edge(nodes[u], nodes[v])
    return new_g
