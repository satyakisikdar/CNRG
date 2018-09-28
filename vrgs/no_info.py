"""
No info extraction and generation

No boundary information is stored.
"""

import networkx as nx
import random
import numpy as np
from copy import deepcopy
from itertools import chain

from vrgs.Rule import NoRule as Rule
from vrgs.globals import find_boundary_edges

def extract_vrg(g, tree, lvl):
    """
    Extract a vertex replacement grammar (specifically an ed-NRC grammar) from a graph given a dendrogram tree
    Stores no boundary info

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
    # non_terminals = set()
    non_terminals = {}  # now a dictionary, key: non-terminal id, val: size of lhs

    new_g = nx.MultiGraph()

    new_g.add_node(0, attr_dict={'label': 0})

    # non_terminals.add(0)
    non_terminals[0] = 0  # non-terminal 0 has size 0

    while len(non_terminals) > 0:  # continue until no more non-terminal nodes
        # choose a non terminal node at random
        sampled_node = random.sample(non_terminals.keys(), 1)[0]
        lhs = non_terminals[sampled_node]

        rhs_candidates = rule_dict[lhs]
        if len(rhs_candidates) == 1:
            rhs = rhs_candidates[0]
        else:
            weights = np.array([rule.frequency for rule in rhs_candidates])
            weights = weights / np.sum(weights)  # normalize into probabilities
            idx = int(np.random.choice(range(len(rhs_candidates)), size=1, p=weights))  # pick based on probability
            rhs = rhs_candidates[idx]

        # find the broken edges
        broken_edges = find_boundary_edges(new_g, [sampled_node])


        assert len(broken_edges) == lhs, "node: {}, expected degree: {}, got: {}".format(sampled_node, lhs, new_g.degree(sampled_node))

        # remove the sampled node
        new_g.remove_node(sampled_node)
        del non_terminals[sampled_node]

        nodes = {}

        for n, d in rhs.graph.nodes_iter(data=True):  # all the nodes are internal
            new_node = node_counter
            nodes[n] = new_node
            new_g.add_node(new_node, attr_dict=d)
            if 'label' in d:  # if it's a new non-terminal add it to the dictionary of non-terminals
                non_terminals[new_node] = d['label']

            node_counter += 1

        # adding the internal edges in the RHS  - no check of effective degree is necessary - WRONG!
        for u, v in rhs.graph.edges_iter():
            new_g.add_edge(nodes[u], nodes[v])

        # randomly assign broken edges to boundary edges
        random.shuffle(broken_edges)

        # possible nonterminals that could be connected to
        possible_nonterminals = set()

        # add to possible candidates the nodes in new_g
        for node in rhs.graph.nodes_iter():
            node = nodes[node]
            if node in non_terminals:   # if terminal, it is a possible candidate by default
                # add only if the non terminal can accommodate more edges
                effective_degree = non_terminals[node] - new_g.degree(node)
                if effective_degree > 0:
                    possible_nonterminals.add(node)


        for u, v in broken_edges:  # either u = node_sample or v is.
            # try the unfulfilled nonterminals first
            if len(possible_nonterminals) > 0:  # try the possible nonterminals
                n = random.sample(possible_nonterminals, 1)[0]
                effective_degree = non_terminals[n] - new_g.degree(n)

                if effective_degree == 1:  # since the effective degree is 1, it cannot take more edges in the future
                    possible_nonterminals.remove(n)

            else:  # try the other internal terminal nodes
                terminal_nodes = {nodes[n]
                                 for n in rhs.graph.nodes_iter()
                                 if nodes[n] not in non_terminals}

                n = random.sample(terminal_nodes, 1)[0]

            if u == sampled_node:
                u = n
            else:
                v = n
            # print('adding boundary edge ({}, {})'.format(u, v))
            new_g.add_edge(u, v)

    return new_g
