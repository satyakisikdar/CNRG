"""
Funky extraction using Justus' method
"""

import networkx as nx
import random
import numpy as np
from copy import deepcopy
import heapq
from collections import defaultdict

import vrgs.globals as globals
from vrgs.Rule import FullRule
from vrgs.Rule import NoRule
from vrgs.Rule import PartRule
from vrgs.globals import find_boundary_edges
from vrgs.part_info import set_boundary_degrees



from vrgs.Tree import create_tree, TreeNode

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
            bucket[val].add(node)   # bucket is a defaultdict
            node2bucket[node] = val

        if node.left is not None:
            stack.append(node.left)

        if node.right is not None:
            stack.append(node.right)

    return nodes, bucket, node2bucket


def extract_subtree(k, buckets, node2bucket, active_nodes):
    """
    :param k:
    :param buckets:
    :param node2bucket:
    :return:
    """

    # pick something from the smallest non-empty bucket

    best_node = None
    for id, bucket in sorted(buckets.items()):
        if len(bucket) != 0:
            best_node = random.sample(bucket, 1)[0]
            break

    if best_node is None:
        return None

    subtree = best_node.payload.intersection(active_nodes)
    new_node_key = min(subtree)

    # print('removing {}, subtree: {}'.format(best_node.key, subtree))

    # best_node is a leaf, so don't add it back to the bucket
    node2bucket[best_node] = None

    # disconnect the children of that node, remove them from active nodes
    stack = [best_node]
    while len(stack) != 0:
        node = stack.pop()

        active_nodes.remove(node.key)
        val = abs(node.nleaf - k)

        if not node.is_leaf:
            buckets[val].remove(node)
            node2bucket[node] = val

        if node.left is not None:
            stack.append(node.left)

        if node.right is not None:
            stack.append(node.right)


    best_node.key = new_node_key  # the best node's key is now the key of the new_node

    active_nodes.add(new_node_key)  # add the new node to the set of active nodes

    best_node.payload = {new_node_key}  # update the payload of the new node
    best_node.left = None
    best_node.right = None
    best_node.is_leaf = True

    if best_node.parent is not None:
        best_node.parent.payload.add(new_node_key)

    # update the nleafs for its parents
    node = best_node

    while node.parent is not None:
        val = node2bucket[node.parent]  # old value of parent
        buckets[val].remove(node.parent)  # remove the parent from that bucket

        node.parent.nleaf -= node.nleaf - 1  # since all the children of the node disappears, but the node remains

        node.parent.payload.add(new_node_key)   # each of the parents has to contain this value
        val = abs(node.parent.nleaf - k)  # new value of parent

        buckets[val].add(node.parent)   # adding the parent to a new bucket
        node2bucket[node.parent] = val   # updating the node2bucket dict

        node = node.parent

    best_node.nleaf = 1   # we can't set this earlier since we are using the value in the while loop

    return subtree

    ## NOTE: nothing is removed from the payload after compression.. always take intersection of payload and active nodes


def funky_extract(g, root,k, mode='full'):
    """
    Runner function for the funcky extract
    :param g: graph
    :param root: pointer to the root of the tree
    :param k: number of leaves to collapse
    :param mode: full / part / no

    :return: list of rules
    """
    nodes, buckets, node2bucket = get_buckets(root=root, k=k)
    active_nodes = {node.key for node in nodes}

    rule_list = list()

    if mode == 'full':
        Rule = FullRule
    elif mode == 'part':
        Rule = PartRule
    else:
        Rule = NoRule

    while True:
        subtree = extract_subtree(k=k, buckets=buckets, node2bucket=node2bucket, active_nodes=active_nodes)
        if subtree is None:
            break

        sg = g.subgraph(subtree)
        boundary_edges = find_boundary_edges(g, subtree)

        rule = Rule()
        rule.lhs = len(boundary_edges)
        rule.internal_nodes = subtree
        # rule.level = lvl

        if mode == 'full':  # in the full information case, we add the boundary edges to the RHS and contract it
            for u, v in boundary_edges:
                sg.add_edge(u, v, attr_dict={'b': True})
            rule.contract_rhs()

        if mode == 'part':  # in the partial boundary info, we need to set the boundary degrees
            set_boundary_degrees(g, sg)

        rule.graph = sg
        rule.generalize_rhs()


        # next we contract the original graph
        [g.remove_node(n) for n in subtree]

        new_node = min(subtree)

        # replace subtree with new_node
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

