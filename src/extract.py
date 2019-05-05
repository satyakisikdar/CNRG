"""
VRG extraction
"""

import logging
import random
from collections import defaultdict
from typing import List, Tuple, Dict, DefaultDict, Set, Any

from tqdm import tqdm

from src.LightMultiGraph import LightMultiGraph
from src.MDL import graph_dl
from src.Rule import FullRule
from src.Rule import NoRule
from src.Rule import PartRule
from src.Tree import TreeNode
from src.VRG import VRG
from src.globals import find_boundary_edges
from src.part_info import set_boundary_degrees


class Record:
    def __init__(self, rule_id: int):
        self.tnodes_list = []
        self.rule_id = rule_id
        self.frequency = 0  # number of times we have seen this rule_id
        self.boundary_edges_list = []
        self.subtree_list = []

    def update(self, boundary_edges, subtree, tnode):
        self.frequency += 1
        self.boundary_edges_list.append(tuple(boundary_edges))
        self.subtree_list.append(tuple(subtree))
        self.tnodes_list.append(tnode)

    def __repr__(self):
        return f'{self.rule_id} > {self.tnodes_list}'

    def __str__(self):
        return f'{self.rule_id} > {self.tnodes_list}'


def node_score_lambda(internal_node: TreeNode, lamb: int) -> int:
    '''
    returns the score of an internal node
    :param internal_node:
    :param lamb:
    :return:
    '''
    return abs(internal_node.nleaf - lamb)


def get_buckets(root: TreeNode, lamb: int) -> Tuple[Set[Any], DefaultDict[float, Any], Dict[float, Any]]:
    """

    :return:
    """
    bucket = defaultdict(set)  # create buckets keyed in by the absolute difference of k and number of leaves and the list of nodes_in_tree
    node2bucket = {}  # keeps track of which bucket every node in the tree is in
    nodes_in_tree = set()  # set of both internal and leaf nodes of the tree
    stack = [root]
    level = {root.key: 0}

    while len(stack) != 0:
        node =  stack.pop()
        nodes_in_tree.add(node)
        score = node_score_lambda(node, lamb)

        if not node.is_leaf:  # don't add to the bucket if it's a leaf
            bucket[score].add(node.key)   # bucket is a defaultdict
            node2bucket[node.key] = score
            for kid in node.kids:
                level[kid.key] = level[node.key] + 1
                kid.level = level[kid.key]
                stack.append(kid)

    return nodes_in_tree, bucket, node2bucket


def update_parents(node: TreeNode, buckets: DefaultDict[float, Any], node2bucket: Dict[Any, int], lamb: int) -> None:
    """

    :param node:
    :param lamb:
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

        new_val = abs(node.parent.nleaf - lamb)  # new value of parent

        buckets[new_val].add(node.parent.key)   # adding the parent to a new bucket
        node2bucket[node.parent.key] = new_val   # updating the node2bucket dict

        node = node.parent
    return


def extract_subtree_original(lamb: int, buckets: DefaultDict[float, Any], node2bucket: Dict[Any, int], active_nodes: Set[Any],
                             key2node: Dict[Any, TreeNode], g: LightMultiGraph, mode: str, grammar: VRG, selection: str) -> Tuple[List[int], Set[int]]:
    """
    :param lamb: size of subtree to be collapsed
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

                    rule, _ = create_rule(subtree=subtree, mode=mode, g=g)  # the rule corresponding to the subtree -

                    if rule in grammar:  # the rule already exists pick that
                        best_node_key = node_key
                        is_existing_rule = True
                        # print('existing rule found!')
                        break  # you dont need to look further

                    if 'mdl' in selection:
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
        update_parents(node=best_node, buckets=buckets, node2bucket=node2bucket, lamb=lamb)

    best_node.make_leaf(new_key=new_node_key)  # make the node a leaf
    return subtree, active_nodes


def create_rule(subtree: List[int], g: LightMultiGraph, mode: str) -> Tuple[PartRule, List[Tuple[int, int]]]:
    sg = g.subgraph(subtree).copy()
    assert isinstance(sg, LightMultiGraph)
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


def compress_graph(g: LightMultiGraph, subtree: List[int], boundary_edges: List[Tuple[int, int]]) -> None:
    '''
    Compress the subtree into one node and return the updated graph
    :param subtree:
    :param boundary_edges:
    :return:
    '''
    subtree = set(subtree)

    # step 1: remove the nodes from subtree
    g.remove_nodes_from(subtree)
    new_node = min(subtree)

    # step 2: replace subtree with new_node
    g.add_node(new_node, label=len(boundary_edges))

    # step 3: rewire new_node
    for u, v in boundary_edges:
        if u in subtree:
            u = new_node
        if v in subtree:
            v = new_node
        g.add_edge(u, v)


def extract_original(g: LightMultiGraph, root: TreeNode, lamb: int, selection: str, mode: str, clustering: str, name: str) -> VRG:
    """
    Runner function for the funcky extract
    :param g: graph
    :param root: pointer to the root of the tree
    :param lamb: number of leaves to collapse
    :param mode: full / part / no

    :return: list of rules
    """
    # start_time = time()

    nodes, buckets, node2bucket = get_buckets(root=root, lamb=lamb)
    active_nodes = {node.key for node in nodes}  # all nodes in the tree
    key2node = {node.key: node for node in nodes}   # key -> node mapping

    tree_nodes_count = len(active_nodes)

    grammar = VRG(mode=mode, lamb=lamb, selection=selection, clustering=clustering, name=name)
    print('Grammar extraction progress')

    with tqdm(total=100, bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]', ncols=50) as pbar:
        while True:
            subtree, active_nodes = extract_subtree_original(lamb=lamb, buckets=buckets, node2bucket=node2bucket, key2node=key2node,
                                                             active_nodes=active_nodes, g=g, mode=mode, grammar=grammar,
                                                             selection=selection)
            # print('Subtree: ', subtree)
            if subtree is None:
                break

            rule, boundary_edges = create_rule(subtree=subtree, mode=mode, g=g)
            assert len(boundary_edges) == rule.lhs

            if rule.lhs == 0 and rule.graph.size() == 0:  # this prevents the dummy start rule
                break
            grammar.add_rule(rule)

            percent = (1 - len(active_nodes) / tree_nodes_count) * 100
            pbar.update(percent - pbar.n)

            compress_graph(g, subtree, boundary_edges)


    # end_time = time()

    # print('\nGrammar extracted in {} secs'.format(round(end_time - start_time, 3)))

    return grammar


def extract_subtree_local(g: LightMultiGraph, root: TreeNode, mode: str, avoid_root: bool) -> Tuple[List[int], PartRule, List[Tuple[int, int]]]:
    '''
    extract the subtree based on mdl
    :param g:
    :param root:
    :param active_nodes: nodes that are presently active
    :return:
    '''
    # TODO: step 1: compute score for each tree node: mdl(G) / (mdl(G | S) + mdl(S))

    active_nodes = set(g.nodes())
    stack: List[TreeNode] = [root]
    best_tnode: TreeNode = None
    best_score = float('-inf')
    best_rule: PartRule = None
    best_boundary_edges: List[Tuple[int, int]] = None

    mdl_g = graph_dl(g)

    while len(stack) != 0:
        g_copy = g.copy()
        tnode =  stack.pop()
        subtree = tnode.leaves & active_nodes

        rule, boundary_edges = create_rule(subtree, g_copy, mode)
        if rule.graph.order() < 2:
            # the rule has less than 2 nodes
            continue
        rule.calculate_cost()
        mdl_s = rule.cost

        compress_graph(g_copy, subtree, boundary_edges)
        mdl_g_s = graph_dl(g_copy)
        score = mdl_g / (mdl_g_s + mdl_s)

        if avoid_root and tnode == root:
            score = float('-inf')

        logging.debug(f'node: {tnode.key}, rule: {rule}, score: {round(score, 3)}')

        if score > best_score:
            best_score = score
            best_tnode = tnode
            best_rule = rule
            best_boundary_edges = boundary_edges

        for kid in tnode.kids:
            if not kid.is_leaf:  # don't add to the bucket if it's a leaf
                stack.append(kid)
    logging.warning(f'best node: {best_tnode}, score: {round(best_score, 3)}')

    # TODO: step 2: pick the node with the max score
    best_subtree = best_tnode.leaves & active_nodes

    # TODO: step 3: update the tree
    best_tnode.make_leaf(new_key=min(best_subtree))

    return best_subtree, best_rule, best_boundary_edges


def extract_local(g: LightMultiGraph, root: TreeNode, selection: str, mode: str, clustering: str, name: str) -> VRG:
    '''
    # do the local MDL thing

    :return:
    '''
    grammar = VRG(mode=mode, selection=selection, clustering=clustering, name=name)
    avoid_root = True

    while True:
        subtree, best_rule, best_boundary_edges = extract_subtree_local(g=g, root=root, mode=mode, avoid_root=avoid_root)
        avoid_root = False
        # print('subtree:', subtree)

        grammar.add_rule(best_rule)
        compress_graph(g, subtree, best_boundary_edges)
        if g.order() == 1:
            break
    return grammar


def update_ancestor_rules(grammar: VRG, g: LightMultiGraph, tnode: TreeNode, tnode_to_record: Dict[TreeNode, Record],
                          mode: str) -> None:
    '''

    :param grammar:
    :param g:
    :param tnode:
    :return:
    '''
    # TODO: 1. get the rule from the grammar - reset

    initial_tnode = tnode
    new_key = min(initial_tnode.leaves)  # this becomes the new leaf node
    leaves = initial_tnode.leaves  # leaf nodes

    children = initial_tnode.children  # children includes all the leaf nodes
    children.add(initial_tnode.key)  # add this to make the updation easier
    children.remove(new_key)  # remove this from the children

    nleaf = initial_tnode.nleaf
    initial_tnode.make_leaf(new_key=new_key)  # make the initial node a leaf

    # TODO: step 1: update the tree data structure
    while tnode.parent is not None:
        tnode.parent.nleaf -= nleaf - 1  # since all the children of the tnode disappears, but the tnode remains
        tnode.parent.leaves -= leaves
        tnode.parent.leaves.add(new_key)
        tnode.parent.children -= children

        tnode = tnode.parent

    # TODO: step 2: update the rules in the grammar using the new subtrees and update the corresponding mapping too
    active_nodes = set(g.nodes())
    tnode = initial_tnode

    while tnode.parent is not None:
        tnode = tnode.parent  # tnode is not its parent
        subtree = tnode.leaves & active_nodes
        # create a new rule
        new_rule, boundary_edges = create_rule(g=g, mode=mode, subtree=subtree)

        # replace existing rule in the grammar with the new rule
        existing_rule_id = tnode_to_record[tnode].rule_id
        existing_rule = grammar.rule_list[existing_rule_id]

        grammar.rule_dict[existing_rule.lhs].remove(existing_rule)
        grammar.rule_list[existing_rule_id] = new_rule
        grammar.rule_dict[new_rule.id].append(new_rule)
        # update tnode_to_record and rule_id_to_records data structures
        # existing_record = tnode_to_record[tnode]

    return


def prune_descendant_rules(grammar: VRG, tnode: TreeNode, tnode_to_record: Dict[TreeNode, Record], rule_id_to_records: Dict[int, Record]) -> None:
    for child in tnode.children:
        if child in tnode_to_record:
            record = tnode_to_record[child]
            rule_id = record.rule_id
            if len(rule_id_to_records[rule_id].tnodes_list) == 1:  # if it's only involved in one rule, we can deactivate safely
                grammar.deactivate_rule(record.rule_id)
                rule = grammar.rule_list[record.rule_id]
                grammar.rule_dict[rule.lhs].remove(rule)


def extract_subtree_global(g: LightMultiGraph, rule_id_to_records: Dict[int, Record], tnode_to_record: Dict[TreeNode, Record],
                           mode: str, grammar: VRG, avoid_root: bool) -> Any:

    # print('node_rule_to_subtrees_boundary_edges:', rule_id_to_records)
    # TODO Step 2: compress the subtrees for each unique rule to find the scores
    best_score = float('-inf')
    best_record_graph = None

    mdl_graph = graph_dl(g)
    for rule_id, record in rule_id_to_records.items():
        rule = grammar.rule_list[rule_id]
        # if not rule.is_active:
        #     print(f'Rule {rule_id} not active')

        g_copy = g.copy()
        rule.calculate_cost()
        mdl_rule = rule.cost

        # compress all subtrees for a rule
        for subtree, boundary_edges in zip(record.subtree_list, record.boundary_edges_list):
            # print(f'{subtree}, {boundary_edges}')
            compress_graph(g=g_copy, boundary_edges=boundary_edges, subtree=subtree)

        mdl_graph_rule = graph_dl(g_copy)

        if 'a' in record.tnodes_list and avoid_root:
            score = float('-inf')
        else:
            score = mdl_graph / (mdl_graph_rule + mdl_rule)
        # score = mdl_graph_rule + mdl_rule

        if rule_id is None:
            print('invalid rule id')
        assert rule_id is not None, 'Invalid rule id'
        logging.debug(f'Rule {rule_id}, score: {round(score, 3)}')
        if rule.id == 3:
            score = 1000

        if score > best_score:
            best_score = score
            best_record_graph = record, g_copy

    #
    best_record, best_graph = best_record_graph
    logging.warning(f'Best rule {best_record.rule_id}, max score: {round(best_score, 3)}')


    # TODO step 4: update the tree - remove descendant grammar rules & update ancestor rules
    for tnode in best_record.tnodes_list:
        prune_descendant_rules(grammar=grammar, tnode=tnode, tnode_to_record=tnode_to_record, rule_id_to_records=rule_id_to_records)
        update_ancestor_rules(grammar=grammar, g=best_graph, tnode=tnode, tnode_to_record=tnode_to_record, mode=mode)

    # TODO step 5: compress the original graph
    for subtree, boundary_edges in zip(best_record.subtree_list, best_record.boundary_edges_list):
        compress_graph(g=g, subtree=subtree, boundary_edges=boundary_edges)

    return grammar


def extract_global(g: LightMultiGraph, root: TreeNode, selection: str, mode: str, clustering: str, name: str) -> VRG:
    '''
    do the global MDL
    '''
    grammar = VRG(mode=mode, selection=selection, clustering=clustering, name=name)

    active_nodes = set(g.nodes())
    stack: List[TreeNode] = [root]

    # TODO step 1: compute the mapping between the internal nodes in the tree and rules
    rule_id_to_records: Dict[Any, Record] = {}  # maps rule_id -> records
    tnode_to_record: Dict[Any, Record] = {}  # maps tree tnode -> rules

    while len(stack) != 0:
        tnode = stack.pop()
        subtree = tnode.leaves & active_nodes

        rule, boundary_edges = create_rule(g=g.copy(), mode=mode, subtree=subtree)
        rule_id = grammar.add_rule(rule)  # gets the rule id

        if rule.graph.order() < 2:
            continue

        if rule_id in rule_id_to_records:  # the rule_id has already been seen before
            rule_id_to_records[rule_id].update(boundary_edges=boundary_edges, subtree=subtree, tnode=tnode)
        else:  # new rule_id
            record = Record(rule_id=rule_id)
            record.update(boundary_edges=boundary_edges, subtree=subtree, tnode=tnode)
            rule_id_to_records[rule_id] = record

        new_rec = Record(rule_id=rule_id)
        new_rec.update(boundary_edges=boundary_edges, subtree=subtree, tnode=tnode)
        tnode_to_record[tnode] = new_rec

        logging.debug(f'tnode: {tnode.key}, rule {rule_id}: {rule}')

        for kid in tnode.kids:
            if not kid.is_leaf:  # don't add to the bucket if it's a leaf
                stack.append(kid)

    avoid_root = True
    while True:
        grammar = extract_subtree_global(g=g, grammar=grammar, mode=mode, rule_id_to_records=rule_id_to_records,
                                         tnode_to_record=tnode_to_record, avoid_root=avoid_root)
        avoid_root = False

        if g.order() == 1:
            break
    # num_nodes = g.order()
    #
    # with tqdm(total=100, bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]', ncols=50) as pbar:
    #     avoid_root = True
    #     while True:
    #         grammar = extract_subtree_global(g, root, mode, grammar, avoid_root)
    #         avoid_root = False  # avoid the root in the first iteration
    #         percent = (1 -  (g.order() - 1)/(num_nodes-1)) * 100
    #         pbar.update(percent - pbar.n)
    #
    #         if g.order() == 1:
    #             break
    #     #     print(f'rule {rule_id}, subtrees: {subtrees}')

    return grammar