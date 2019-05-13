"""
Partial info extraction and generation

Partial boundary information containing node level info on boundary degree
"""

import random

import networkx as nx
import numpy as np

from src.LightMultiGraph import LightMultiGraph
from src.globals import find_boundary_edges


def set_boundary_degrees(g, sg):
    # TODO: test this!!
    boundary_degree = {n: 0 for n in sg.nodes()}  # by default every boundary degree is 0

    for u, v in nx.edge_boundary(g, sg.nodes()):
        if sg.has_node(u):
            boundary_degree[u] += g.number_of_edges(u, v)
        else:
            boundary_degree[v] += g.number_of_edges(u, v)
    nx.set_node_attributes(sg, values=boundary_degree, name='b_deg')

def set_boundary_degrees_old(g, sg):
    """
    Find the nunber of boundary edges that each node participate in.
    This is stored as a node level attribute - 'b_deg' in nodes in g that are part of nbunch

    :param g: whole graph
    :param sg: the subgraph
    :return: nothing
    """
    boundary_degree = {}

    for u in sg.nodes():
        boundary_degree[u] = 0
        for v in g.neighbors(u):
            if not sg.has_node(v):
                boundary_degree[u] += g.number_of_edges(u, v)   # for a multi-graph

    nx.set_node_attributes(sg, values=boundary_degree, name='b_deg')


def generate_graph(rule_dict, rule_list):
    """
    Create a new graph from the VRG at random
    :param rule_dict: List of unique VRG rules
    :return: newly generated graph
    """

    node_counter = 1
    non_terminals = set()
    # new_g = nx.MultiGraph()
    new_g = LightMultiGraph()

    new_g.add_node(0, label=0)
    non_terminals.add(0)

    rule_ordering = []  # list of rule ids in the order they were fired

    while len(non_terminals) > 0:      # continue until no more non-terminal nodes
        # choose a non terminal node at random
        node_sample = random.sample(non_terminals, 1)[0]
        lhs = new_g.nodes[node_sample]['label']

        rhs_candidates = list(filter(lambda rule: rule.is_active, rule_dict[lhs]))
        # consider only active rules

        if len(rhs_candidates) == 1:
            rhs = rhs_candidates[0]
        else:
            weights = np.array([rule.frequency for rule in rhs_candidates])
            weights = weights / np.sum(weights)   # normalize into probabilities
            idx = int(np.random.choice(range(len(rhs_candidates)), size=1, p=weights))  # pick based on probability
            rhs = rhs_candidates[idx]

        # print(f'firing rule {rule_list.index(rhs)}')
        # rule_ordering.append(rule_list.index(rhs))
        # print('Selected node {} with label {}'.format(node_sample, lhs))

        broken_edges = find_boundary_edges(new_g, [node_sample])

        # print('broken edges: ', broken_edges)

        assert len(broken_edges) == lhs

        new_g.remove_node(node_sample)
        non_terminals.remove(node_sample)

        nodes = {}

        for n, d in rhs.graph.nodes(data=True):   # all the nodes are internal
            new_node = node_counter
            nodes[n] = new_node
            new_g.add_node(new_node, attr_dict=d)
            if 'label' in d:  # if it's a new non-terminal add it to the set of non-terminals
                non_terminals.add(new_node)
            node_counter += 1


        # randomly assign broken edges to boundary edges
        random.shuffle(broken_edges)

        # randomly joining the new boundary edges from the RHS to the rest of the graph - uniformly at random
        for n, d in rhs.graph.nodes(data=True):
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
        for u, v in rhs.graph.edges():
            # print('adding RHS internal edge ({}, {})'.format(nodes[u], nodes[v]))
            edge_multiplicity = rhs.graph[u][v]['weight']  #
            for _ in range(edge_multiplicity):
                new_g.add_edge(nodes[u], nodes[v])
    return new_g, rule_ordering


if __name__ == '__main__':
    g = LightMultiGraph()
    g.add_edges_from([(1, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
    sg = g.subgraph([2, 3]).copy()
    print(g.edges(data=True))
    set_boundary_degrees(g, sg)
    print(sg.nodes(data=True))

