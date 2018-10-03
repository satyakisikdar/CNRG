"""
Funky extraction using Justus' method
"""

import random
from collections import defaultdict

from vrgs.Rule import FullRule
from vrgs.Rule import NoRule
from vrgs.Rule import PartRule
from vrgs.globals import find_boundary_edges
from vrgs.part_info import set_boundary_degrees


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


def extract_subtree(k, buckets, node2bucket, active_nodes, key2node):
    """
    :param k: size of subtree to be collapsed
    :param buckets: mapping of val -> set of internal nodes in the tree
    :param node2bucket: mapping of key -> node
    :return: subtree, set of active nodes
    """

    # pick something from the smallest non-empty bucket
    pick_randomly = True
    best_node = None

    for _, bucket in sorted(buckets.items()):
        if pick_randomly and len(bucket) != 0:   # if picking randomly, and the bucket is not empty
            possible_nodes = bucket & active_nodes   # possible nodes are the ones that are allowed
            if len(possible_nodes) == 0:
                continue

            best_node_key = random.sample(possible_nodes, 1)[0]  # sample 1 node from the min bucket randomly
            best_node = key2node[best_node_key]

            bucket.remove(best_node_key)  # remove the best node from the bucket
            break

    if best_node is None:
        return None, None

    subtree = best_node.leaves & active_nodes   # only consider the leaves that are active

    new_node_key = min(subtree)  # key of the new node

    active_nodes = active_nodes - best_node.children  # all the children of this node are no longer active
    active_nodes.remove(best_node.key)  # remove the old node too
    active_nodes.add(new_node_key)  # add the new node key
    best_node.key = new_node_key  # update the key

    if best_node.parent is not None:
        update_parents(node=best_node, buckets=buckets, node2bucket=node2bucket, k=k)

    best_node.make_leaf(new_key=new_node_key)  # make the node a leaf

    return subtree, active_nodes


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
    active_nodes = {node.key for node in nodes}  # all nodes in the tree
    key2node = {node.key: node for node in nodes}   # key -> node mapping

    rule_list = list()

    if mode == 'full':
        Rule = FullRule

    elif mode == 'part':
        Rule = PartRule

    else:
        Rule = NoRule

    while True:
        subtree, active_nodes = extract_subtree(k=k, buckets=buckets, node2bucket=node2bucket, active_nodes=active_nodes, key2node=key2node)
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
        for u, v in boundary_edges:
            if u in subtree:
                u = new_node
            if v in subtree:
                v = new_node
            g.add_edge(u, v)

        rule_list.append(rule)  # add the rule into the rule list

    return rule_list
