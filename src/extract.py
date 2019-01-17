"""
VRG extraction
"""

import random
from collections import defaultdict
import networkx as nx
import numpy as np
from time import time
from copy import deepcopy

from src.Rule import FullRule
from src.Rule import NoRule
from src.Rule import PartRule
from src.globals import find_boundary_edges
from src.part_info import set_boundary_degrees
from src.VRG import VRG


def get_buckets(root, k):
    """

    :return:
    """
    bucket = defaultdict(set)  # create buckets keyed in by the absolute difference of k and number of leaves and the list of nodes
    node2bucket = {}  # keeps track of which bucket every node in the tree is in
    nodes = set()
    stack = [root]
    level = {root.key: 0}

    while len(stack) != 0:
        node =  stack.pop()
        nodes.add(node)
        val = abs(node.nleaf - k)

        if not node.is_leaf:  # don't add to the bucket if it's a leaf
            bucket[val].add(node.key)   # bucket is a defaultdict
            node2bucket[node.key] = val
            for kid in node.kids:
                level[kid.key] = level[node.key] + 1
                kid.level = level[kid.key]
                stack.append(kid)

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


def extract_subtree(k, buckets, node2bucket, active_nodes, key2node, g, mode, grammar, selection):
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

    if selection == 'mdl':
        min_key = float('inf')  # min MDL
    elif selection == 'level':
        min_key = float('inf')  # min level
    elif selection == 'level_mdl':
        min_key = (float('inf'), float('inf'))  # min_level, min_MDL
    else:  # random
        min_key = None  # pick randomly


    for _, bucket in sorted(buckets.items()):  # start with the bucket with the min value
        possible_node_keys = bucket & active_nodes  # possible nodes are the ones that are allowed

        if len(possible_node_keys) == 0:  # if there is no possible nodes, move on
            continue

        elif len(possible_node_keys) == 1:  # just one node in the bucket
            best_node_key = possible_node_keys.pop()

        else:  # the bucket has more than one node


            if selection not in ('mdl', 'level', 'level_mdl'):    # if picking randomly, and the bucket is not empty
                best_node_key = random.sample(possible_node_keys, 1)[0]  # sample 1 node from the min bucket randomly

            else:  # rule selection is non random
                is_existing_rule = False  # checks if the bucket contains any existing rules
                min_node_key = None

                for node_key in possible_node_keys:
                    subtree = key2node[node_key].leaves & active_nodes  # only consider the leaves that are active
                    rule_level = key2node[node_key].level

                    assert isinstance(rule_level, int), 'rule level not int'

                    rule, _ = create_rule(subtree=subtree, mode=mode, g=g)  # the rule corresponding to the subtree - TODO: graph g gets mutated - stop that

                    if rule in grammar:  # the rule already exists pick that
                        best_node_key = node_key
                        is_existing_rule = True
                        # print('existing rule found!')
                        break  # you dont need to look further

                    rule.calculate_cost()  # find the MDL cost of the rule
                    # print('node: {} mdl: {}'.format(node_key, rule.cost))

                    if selection == 'mdl':
                        if rule.cost < min_key:
                            min_key = rule.cost
                            min_node_key = node_key

                    elif selection == 'level':
                        if rule_level < min_key:
                            min_key = rule_level
                            min_node_key = node_key

                    else:   # selection is level_mdl
                        if (rule_level, rule.cost) < min_key:
                            min_key = (rule_level, rule.cost)
                            min_node_key = node_key

                if not is_existing_rule:  # all the rules were new
                    best_node_key = min_node_key

        # print('Picking node {} from the bucket level {}'.format(best_node_key, min_key[1]))
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
        rule = FullRule(lhs=len(boundary_edges), internal_nodes=subtree, graph=sg)

        for u, v in boundary_edges:
            rule.graph.add_edge(u, v, attr_dict={'b': True})

        rule.contract_rhs()  # contract and generalize

    elif mode == 'part':  # in the partial boundary info, we need to set the boundary degrees
        rule = PartRule(lhs=len(boundary_edges), graph=sg)
        set_boundary_degrees(g, rule.graph)
        rule.generalize_rhs()

    else:
        rule = NoRule(lhs=len(boundary_edges), graph=sg)
        rule.generalize_rhs()
    return rule, boundary_edges


def extract(g, root, k, selection, mode, clustering):
    """
    Runner function for the funcky extract
    :param g: graph
    :param root: pointer to the root of the tree
    :param k: number of leaves to collapse
    :param mode: full / part / no

    :return: list of rules
    """

    # start_time = time()

    nodes, buckets, node2bucket = get_buckets(root=root, k=k)
    active_nodes = {node.key for node in nodes}  # all nodes in the tree
    key2node = {node.key: node for node in nodes}   # key -> node mapping

    grammar = VRG(mode=mode, k=k, selection=selection, clustering=clustering)

    while True:
        subtree, active_nodes = extract_subtree(k=k, buckets=buckets, node2bucket=node2bucket, key2node=key2node,
                                                active_nodes=active_nodes, g=g, mode=mode, grammar=grammar,
                                                selection=selection)
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

    # end_time = time()

    # print('\nGrammar extracted in {} secs'.format(round(end_time - start_time, 3)))

    return grammar