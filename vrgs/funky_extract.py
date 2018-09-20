"""
Funky extraction using Justus' method
"""

import networkx as nx
import random
import numpy as np
from copy import deepcopy
import heapq

import vrgs.globals as globals
from vrgs.Rule import FullRule
# from vrgs.Rule import NoRule
# from vrgs.Rule import PartlRule
from vrgs.globals import find_boundary_edges

from collections import defaultdict

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


def extract_vrg(root, k, nodes, bucket, node2bucket):
    """
    :param k:
    :param bucket:
    :param node2bucket:
    :return:
    """
    active_nodes = {node.key for node in nodes}
    new_node_key = max(filter(lambda x: isinstance(x, int), active_nodes))
    best_node = None
    while best_node != root:
        # pick something from the smallest non-empty bucket
        min_bucket = 0
        while True:
            best_node = None
            for node in bucket[min_bucket]:  # look for nodes in the min_bucket
                best_node = node
                break

            if best_node is not None:
                break
            else:
                min_bucket += 1

        subtree = best_node.payload.intersection(active_nodes)

        print('removing {}, subtree: {}'.format(best_node.key, subtree))

        # best_node is a leaf, so don't add it back to the bucket
        node2bucket[best_node] = None

        # disconnect the children of that node, remove them from active nodes
        stack = [best_node]
        while len(stack) != 0:
            node = stack.pop()

            active_nodes.remove(node.key)
            val = abs(node.nleaf - k)

            if not node.is_leaf:
                bucket[val].remove(node)
                node2bucket[node] = val

            if node.left is not None:
                stack.append(node.left)

            if node.right is not None:
                stack.append(node.right)


        new_node_key += 1 # the best node in the tree is replaced by this new node
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
            bucket[val].remove(node.parent)  # remove the parent from that bucket

            node.parent.nleaf -= node.nleaf - 1 # since all the children of the node disappears, but the node remains

            node.parent.payload.add(new_node_key)   # each of the parents has to contain this value
            val = abs(node.parent.nleaf - k)  # new value of parent

            bucket[val].add(node.parent)   # adding the parent to a new bucket
            node2bucket[node.parent] = val   # updating the node2bucket dict

            node = node.parent

        best_node.nleaf = 1   # we can't set this earlier since we are using the value in the while loop


    ## NOTE: nothing is removed from the payload after compression.. always take intersection of payload and active nodes


def funky_runner(root,k):
    """
    Runner function for the funcky extract
    :param root: pointer to the root of the tree
    :return: list of rules
    """
    nodes, bucket, node2bucket = get_buckets(root=root, k=k)
    extract_vrg(root=root, k=k, bucket=bucket, node2bucket=node2bucket, nodes=nodes)


#
# k = 6
# root = create_tree()
# nodes, bucket, node2bucket = get_buckets(root=root, k=k)
# extract_vrg(root=root, k=k, bucket=bucket, node2bucket=node2bucket, nodes=nodes)

