"""
Partial info extraction and generation

Partial boundary information containing node level info on boundary degree
"""

import networkx as nx
import random
import numpy as np
import vrgs.globals as globals
from vrgs.Rule import Rule

def set_boundary_degrees(g, sg):
    """
    Find the nunber of boundary edges that each node participate in.
    This is stored as a node level attribute - 'b_deg' in nodes in g that are part of nbunch

    :param g: whole graph
    :param sg: the subgraph
    :return: nothing
    """
    boundary_degree = {}

    for u in sg.nodes_iter():
        boundary_degree[u] = 0
        for v in g.neighbors_iter(u):
            if not sg.has_node(v):
                boundary_degree[u] += g.number_of_edges(u, v)   # for a multi-graph

    nx.set_node_attributes(sg, 'b_deg', boundary_degree)


def find_boundary_edges(g, nbunch):
    """
    Collect all of the boundary edges (i.e., the edges
    that connect the subgraph to the original graph)

    :param g: whole graph
    :param nbunch: set of nodes in the subgraph
    :return: boundary edges
    """
    nbunch = set(nbunch)
    boundary_edges = []
    for u, v in g.edges_iter():
        if u in nbunch and v not in nbunch:
            boundary_edges.append((u, v))
        elif u not in nbunch and v in nbunch:
            boundary_edges.append((u, v))
    return boundary_edges


def extract_vrg(g, tree, lvl):
    """
    Extract a vertex replacement grammar (specifically an ed-NRC grammar) from a graph given a dendrogram tree
    Stores only partial boundary information, specifically only the boundary degree

    :param g: graph to extract from
    :param tree: dendrogram with nodes at the bottom.
    :param lvl: level of discovery - root at level 0
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
        set_boundary_degrees(g, sg)
        boundary_edges = find_boundary_edges(g, subtree)

        # print('st:', subtree, nbunch)

        rule = Rule()
        rule.lhs = len(boundary_edges)
        rule.graph = sg
        rule.level = lvl
        rule.internal_nodes = subtree
        rule.generalize_rhs()
        # rule.contract_rhs()

        # next we contract the original graph
        [g.remove_node(n) for n in subtree]

        new_node = min(subtree)

        # replace subtree with new_node
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

        rule_list.append(rule)
    return rule_list

def generate_graph(rule_dict):
    """
    Create a new graph from the VRG at random
    :param rule_dict: List of unique VRG rules
    :return: newly generated graph
    """
    node_counter = 1
    non_terminals = set()
    new_g = nx.MultiGraph()

    new_g.add_node(0, attr_dict={'label': 0})
    non_terminals.add(0)

    while len(non_terminals) > 0:      # continue until no more non-terminal nodes
        # choose a non terminal node at random
        node_sample = random.sample(non_terminals, 1)[0]
        lhs = new_g.node[node_sample]['label']

        rhs_candidates = rule_dict[lhs]
        if len(rhs_candidates) == 1:
            rhs = rhs_candidates[0]
        else:
            weights = np.array([rule.frequency for rule in rhs_candidates])
            weights = weights / np.sum(weights)   # normalize into probabilities
            idx = int(np.random.choice(range(len(rhs_candidates)), size=1, p=weights))  # pick based on probability
            rhs = rhs_candidates[idx]

        # print('Selected node {} with label {}'.format(node_sample, lhs))

        broken_edges = find_boundary_edges(new_g, [node_sample])

        # print('broken edges: ', broken_edges)

        assert len(broken_edges) == lhs

        new_g.remove_node(node_sample)
        non_terminals.remove(node_sample)

        nodes = {}

        for n, d in rhs.graph.nodes_iter(data=True):   # all the nodes are internal
            new_node = node_counter
            nodes[n] = new_node
            new_g.add_node(new_node, attr_dict=d)
            if 'label' in d:  # if it's a new non-terminal add it to the set of non-terminals
                non_terminals.add(new_node)
            node_counter += 1


        # randomly assign broken edges to boundary edges
        random.shuffle(broken_edges)

        # randomly joining the new boundary edges from the RHS to the rest of the graph - uniformly at random
        for n, d in rhs.graph.nodes_iter(data=True):
            num_boundary_edges = d['b_deg']
            if num_boundary_edges == 0:  # there are no boundary edges incident to that node
                continue

            assert len(broken_edges) >= num_boundary_edges

            edge_candidates = broken_edges[: num_boundary_edges]   # picking the first num_broken edges
            broken_edges = broken_edges[num_boundary_edges: ]    # removing them from future consideration

            for u, v in edge_candidates:  # each edge is either (node_sample, v) or (u, node_sample)
                if u == node_sample:
                    u = nodes[n]
                else:
                    v = nodes[n]
                # print('adding broken edge ({}, {})'.format(u, v))
                new_g.add_edge(u, v)


        # adding the rhs to the new graph
        for u, v in rhs.graph.edges_iter():
            # print('adding RHS internal edge ({}, {})'.format(nodes[u], nodes[v]))
            new_g.add_edge(nodes[u], nodes[v])
    return new_g

