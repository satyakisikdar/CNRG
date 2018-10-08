"""
Funky extraction using Justus' method
"""

import random
from collections import defaultdict
import networkx as nx
import numpy as np
from time import time

from vrgs.Rule import FullRule
from vrgs.Rule import NoRule
from vrgs.Rule import PartRule
from vrgs.globals import find_boundary_edges
from vrgs.part_info import set_boundary_degrees
from vrgs.VRG import VRG

def get_buckets(root, k):
    """

    :return:
    """
    bucket = defaultdict(set)  # create buckets keyed in by the absolute difference of k and number of leaves and the list of nodes
    node2bucket = {}  # keeps track of which bucket every node in the tree is in
    nodes = set()
    stack = [root]

    while len(stack) != 0:
        node =  stack.pop()
        nodes.add(node)
        val = abs(node.nleaf - k)

        if not node.is_leaf:  # don't add to the bucket if it's a leaf
            bucket[val].add(node.key)   # bucket is a defaultdict
            node2bucket[node.key] = val

        if node.left is not None:
            stack.append(node.left)

        if node.right is not None:
            stack.append(node.right)

    return nodes, bucket, node2bucket


def update_parents(node, buckets, node2bucket, k):
    """

    :param node:
    :param k:
    :return:
    """
    # the subtree that is removed to be removed
    node_key = node.key
    subtree = node.leaves
    subtree.remove(node.key)

    while node.parent is not None:
        old_val = node2bucket[node.parent.key]  # old value of parent
        buckets[old_val].remove(node.parent.key)  # remove the parent from that bucket

        node.parent.nleaf -= node.nleaf - 1  # since all the children of the node disappears, but the node remains
        node.parent.leaves -= subtree
        node.parent.leaves.add(node_key)

        new_val = abs(node.parent.nleaf - k)  # new value of parent

        buckets[new_val].add(node.parent.key)   # adding the parent to a new bucket
        node2bucket[node.parent.key] = new_val   # updating the node2bucket dict

        node = node.parent
    return


def extract_subtree(k, buckets, node2bucket, active_nodes, key2node, g, mode, grammar, pick_randomly=True):
    """
    :param k: size of subtree to be collapsed
    :param buckets: mapping of val -> set of internal nodes in the tree
    :param node2bucket: mapping of key -> node
    :param g: graph
    :return: subtree, set of active nodes
    """

    # pick something from the smallest non-empty bucket
    best_node_key = None  # the key of the best node found from the buckets
    best_node = None

    for _, bucket in sorted(buckets.items()):  # start with the bucket with the min value
        possible_node_keys = bucket & active_nodes  # possible nodes are the ones that are allowed

        if len(possible_node_keys) == 0:  # if there is no possible nodes, move on
            continue

        elif len(possible_node_keys) == 1:  # just one node in the bucket
            best_node_key = possible_node_keys.pop()

        else:  # the bucket has more than one nodes
            if pick_randomly:    # if picking randomly, and the bucket is not empty
                best_node_key = random.sample(possible_node_keys, 1)[0]  # sample 1 node from the min bucket randomly

            else:  # rule selection is MDL based
                is_existing_rule = False  # checks if the bucket contains any existing rules
                min_mdl = float('inf')
                min_mdl_node_key = None

                for node_key in possible_node_keys:
                    subtree = key2node[node_key].leaves & active_nodes  # only consider the leaves that are active
                    rule, _ = create_rule(subtree=subtree, mode=mode, g=g)  # the rule corresponding to the subtree

                    if rule in grammar:  # the rule already exists pick that
                        best_node_key = node_key
                        is_existing_rule = True
                        print('existing rule found!')
                        break  # you dont need to look further

                    rule.calculate_cost()  # find the MDL cost of the rule

                    if rule.cost < min_mdl:  # keeps track of the rule with the minimum MDL in case there are no existing rules
                        min_mdl = rule.cost
                        min_mdl_node_key = node_key

                if not is_existing_rule:  # all the rules were new
                    best_node_key = min_mdl_node_key

        # print('Picking node {} from the bucket'.format(best_node_key))
        best_node = key2node[best_node_key]
        break

    if best_node is None:
        return None, None

    subtree = best_node.leaves & active_nodes  # only consider the leaves that are active

    new_node_key = min(subtree)  # key of the new node

    active_nodes = active_nodes - best_node.children  # all the children of this node are no longer active
    active_nodes.remove(best_node.key)  # remove the old node too
    active_nodes.add(new_node_key)  # add the new node key
    best_node.key = new_node_key  # update the key

    if best_node.parent is not None:
        update_parents(node=best_node, buckets=buckets, node2bucket=node2bucket, k=k)

    best_node.make_leaf(new_key=new_node_key)  # make the node a leaf

    return subtree, active_nodes


def create_rule(subtree, g, mode):
    sg = g.subgraph(subtree)
    boundary_edges = find_boundary_edges(g, subtree)

    if mode == 'full':  # in the full information case, we add the boundary edges to the RHS and contract it
        rule = FullRule()
        rule.lhs = len(boundary_edges)
        rule.internal_nodes = subtree

        for u, v in boundary_edges:
            sg.add_edge(u, v, attr_dict={'b': True})

        rule.graph = sg
        rule.contract_rhs()  # TODO breaks generation - fix this.. - currently commented out in FullRule
        rule.generalize_rhs()

    elif mode == 'part':  # in the partial boundary info, we need to set the boundary degrees
        rule = PartRule()
        rule.lhs = len(boundary_edges)
        rule.internal_nodes = subtree

        set_boundary_degrees(g, sg)

        rule.graph = sg
        rule.generalize_rhs()

    else:
        rule = NoRule()
        rule.lhs = len(boundary_edges)
        rule.internal_nodes = subtree
        rule.graph = sg
        rule.generalize_rhs()
    return rule, boundary_edges


def funky_extract(g, root,k, pick_randomly, mode):
    """
    Runner function for the funcky extract
    :param g: graph
    :param root: pointer to the root of the tree
    :param k: number of leaves to collapse
    :param mode: full / part / no

    :return: list of rules
    """

    start_time = time()

    nodes, buckets, node2bucket = get_buckets(root=root, k=k)
    active_nodes = {node.key for node in nodes}  # all nodes in the tree
    key2node = {node.key: node for node in nodes}   # key -> node mapping

    grammar = VRG(mode=mode, k=k)

    while True:
        subtree, active_nodes = extract_subtree(k=k, buckets=buckets, node2bucket=node2bucket, key2node=key2node,
                                                active_nodes=active_nodes, g=g, mode=mode, grammar=grammar,
                                                pick_randomly=pick_randomly)
        if subtree is None:
            break

        rule, boundary_edges = create_rule(subtree=subtree, mode=mode, g=g)
        grammar.add_rule(rule)

        # next we contract the original graph
        [g.remove_node(n) for n in subtree]

        new_node = min(subtree)

        assert len(boundary_edges) == rule.lhs

        # replace subtree with new_node
        g.add_node(new_node, attr_dict={'label': rule.lhs})

        # rewire new_node
        for u, v in boundary_edges:
            if u in subtree:
                u = new_node
            if v in subtree:
                v = new_node
            g.add_edge(u, v)

    end_time = time()

    print('\nGrammar extracted in {} secs'.format(round(end_time - start_time, 3)))

    return grammar


def funky_generate(grammar, mode):
    """
    Generate a graph using the grammar
    :param grammar: a VRG object
    :param mode: full / part / no
    :return:
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

        rhs_candidates = grammar.rule_dict[lhs]
        if len(rhs_candidates) == 1:
            rhs = rhs_candidates[0]
        else:
            weights = np.array([rule.frequency for rule in rhs_candidates])
            weights = weights / np.sum(weights)   # normalize into probabilities
            idx = int(np.random.choice(range(len(rhs_candidates)), size=1, p=weights))  # pick based on probability
            rhs = rhs_candidates[idx]

        if mode == 'full':  # only full information has boundary node info
            max_v = -1
            for v in rhs.graph.nodes_iter():
                if isinstance(v, int):
                    max_v = max(v, max_v)
            max_v += 1

            # expanding the 'I' nodes into separate integer labeled nodes
            if rhs.graph.has_node('I'):
                for u, v in rhs.graph.edges():
                    if u == 'I':
                        rhs.graph.remove_edge(u, v)
                        rhs.graph.add_edge(max_v, v, attr_dict={'b': True})
                        max_v += 1
                    elif v == 'I':
                        rhs.graph.remove_edge(u, v)
                        rhs.graph.add_edge(u, max_v, attr_dict={'b': True})
                        max_v += 1

                assert rhs.graph.degree('I') == 0
                rhs.graph.remove_node('I')

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


    return new_g