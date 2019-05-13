"""
Full info extraction and generation

Uses explicit boundary information containing node level info on boundary nodes and edges
"""

import random

import networkx as nx
import numpy as np

from src.globals import find_boundary_edges


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


def generate_graph(rule_dict, rule_list):
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

    rule_ordering = []

    while len(non_terminals) > 0:      # continue until no more non-terminal nodes exist
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

        rule_ordering.append(rule_list.index(rhs))

        max_v = -1
        for v in rhs.graph.nodes_iter():
            if isinstance(v, int):
                max_v = max(v, max_v)
        max_v += 1

        # expanding the 'Iso' nodes into separate integer labeled nodes
        if rhs.graph.has_node('Iso'):
            for u, v in rhs.graph.edges():
                if u == 'I':
                    rhs.graph.remove_edge(u, v)
                    rhs.graph.add_edge(max_v, v, attr_dict={'b': True})
                    max_v += 1

                elif v == 'Iso':
                    rhs.graph.remove_edge(u, v)
                    rhs.graph.add_edge(u, max_v, attr_dict={'b': True})
                    max_v += 1

            assert rhs.graph.degree('Iso') == 0
            rhs.graph.remove_node('Iso')

        broken_edges = find_boundary_edges(new_g, [node_sample])

        assert len(broken_edges) == lhs, 'expected {}, got {}'.format(lhs, len(broken_edges))

        new_g.remove_node(node_sample)
        non_terminals.remove(node_sample)

        nodes = {}

        for n, d in rhs.graph.nodes_iter(data=True):
            if isinstance(n, str):
                new_node = node_counter
                nodes[n] = new_node
                new_g.add_node(new_node, attr_dict=d)
                if 'label' in d:  # if it's a new non-terminal add it to the set of non-terminals
                    non_terminals.add(new_node)
                node_counter += 1

        for u, v, d in rhs.graph.edges_iter(data=True):
            if 'b' not in d:  # (u, v) is not a boundary edge
                  new_g.add_edge(nodes[u], nodes[v])

        # randomly assign broken edges to boundary edges
        random.shuffle(broken_edges)

        boundary_edge_count = 0
        for u, v,  d in rhs.graph.edges_iter(data=True):
            if 'b' in d:  # (u, v) is a boundary edge
                boundary_edge_count += 1

        assert len(broken_edges) >= boundary_edge_count, 'broken edges {}, boundary edges {}'.format(len(broken_edges),
                                                                                                    boundary_edge_count)
        for u, v,  d in rhs.graph.edges_iter(data=True):
            if 'b' not in d:  # (u, v) is not a boundary edge
                continue

            b_u, b_v = broken_edges.pop()
            if isinstance(u, str):  # u is internal
                if b_u == node_sample:  # b_u is the sampled node
                    new_g.add_edge(nodes[u], b_v)
                else:
                    new_g.add_edge(nodes[u], b_u)
            else:  # v is internal
                if b_u == node_sample:
                    new_g.add_edge(nodes[v], b_v)
                else:
                    new_g.add_edge(nodes[v], b_u)

    return new_g, rule_ordering
