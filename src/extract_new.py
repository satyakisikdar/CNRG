"""
VRG extraction
"""
import abc
import logging
import random
from collections import defaultdict
from typing import List, Tuple, Dict, DefaultDict, Set, Any, Union, Optional
import math
from time import time

from tqdm import tqdm

from src.LightMultiGraph import LightMultiGraph
from src.MDL import graph_dl
from src.Rule import FullRule, NoRule, PartRule
from src.globals import find_boundary_edges
from src.part_info import set_boundary_degrees

from src.Tree_new import TreeNodeNew
from src.VRG_new import VRG

class Record:
    __slots__ = 'tnodes_list', 'rule_id', 'frequency', 'boundary_edges_list', 'subtree_list', 'score'

    def __init__(self, rule_id: int):
        self.tnodes_list: List[TreeNodeNew] = []
        self.rule_id: int = rule_id
        self.frequency: int = 0  # number of times we have seen this rule_id
        self.boundary_edges_list: List[Set[Tuple[int, int]]] = []
        self.subtree_list: List[Set[int]] = []
        self.score = None  # score of the rule

    def update(self, boundary_edges: Any, subtree: Set[int], tnode: TreeNodeNew):
        self.frequency += 1
        self.boundary_edges_list.append(tuple(boundary_edges))
        self.subtree_list.append(tuple(subtree))
        self.tnodes_list.append(tnode)

    def remove(self):
        self.frequency -= 1

    def __repr__(self):
        st = ''
        if self.frequency == 0:
            st += '[x] '
        st +=  f'{self.rule_id} > {self.tnodes_list}'
        return st

    def __str__(self):
        st = ''
        if self.frequency == 0:
            st += '[x] '
        st += f'{self.rule_id} > {self.tnodes_list} {round(self.score, 3)}'
        return st


def create_rule(subtree: Set[int], g: LightMultiGraph, mode: str) -> Tuple[PartRule, List[Tuple[int, int]]]:
    sg = g.subgraph(subtree).copy()
    assert isinstance(sg, LightMultiGraph)
    boundary_edges = find_boundary_edges(g, subtree)

    if mode == 'full':  # in the full information case, we add the boundary edges to the RHS and contract it
        rule = FullRule(lhs=len(boundary_edges), internal_nodes=subtree, graph=sg)

        for u, v in boundary_edges:
            rule.graph.add_edge(u, v, b=True)

        rule.contract_rhs()  # contract and generalize

    elif mode == 'part':  # in the partial boundary info, we need to set the boundary degrees
        rule = PartRule(lhs=len(boundary_edges), graph=sg)
        set_boundary_degrees(g, rule.graph)
        rule.generalize_rhs()

    else:
        rule = NoRule(lhs=len(boundary_edges), graph=sg)
        rule.generalize_rhs()
    return rule, boundary_edges


def compress_graph(g: LightMultiGraph, subtree: Set[int], boundary_edges: Any) -> None:
    """
    Compress the subtree into one node and return the updated graph
    :param subtree:
    :param boundary_edges:
    :return:
    """
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

    # TODO: reverse this to reconstruct the original graph - keep track of the edges being added and the new node - this might be faster than copying

class BaseExtractor(abc.ABC):
    # __slots__ = 'type', 'g', 'root', 'tnode_to_score', 'grammar', 'mu'

    def __init__(self, g:LightMultiGraph, type: str, root: TreeNodeNew, grammar: VRG, mu: int) -> None:
        assert type in ('local_dl', 'global_dl', 'mu_random', 'mu_level', 'mu_dl', 'mu_level_dl'), f'Invalid mode: {type}'
        self.type = type
        self.g = g  # the graph
        self.root = root
        self.tnode_to_score: Dict[TreeNodeNew, Any] = {}
        self.grammar = grammar
        self.mu = mu

    def __str__(self) -> str:
        st = f'Type: {self.type}, mu: {self.mu}'
        return st

    def __repr__(self) -> str:
        return str(self)

    def get_sorted_tnodes(self):
        tnodes, scores = zip(*sorted(self.tnode_to_score.items(), key=lambda kv: kv[1]))
        return tnodes

    def get_best_tnode_and_score(self) -> Any:
        """
        returns the tnode with the lowest score
        :return: tnode
        """
        return min(self.tnode_to_score.items(), key=lambda kv: kv[1])  # use the value as the key

    def update_subtree_scores(self, start_tnode: TreeNodeNew) -> Any:
        """
        updates scores of the tnodes of the subtree rooted at start_tnode depending on the extraction type
        :param start_tnode: starting tnode. for the entire tree, use self.root
        :return:
        """
        active_nodes = set(self.g.nodes())
        stack: List[TreeNodeNew] = [start_tnode]

        is_global_extractor = hasattr(self, 'rule_id_to_record')

        while len(stack) != 0:
            tnode = stack.pop()
            subtree = tnode.leaves & active_nodes

            if is_global_extractor:  # this is true for GlobalExtractor objects only
                self.rule_id_to_record: Dict[int, Record]

                rule, boundary_edges = create_rule(subtree=subtree, g=self.g, mode='part')
                rule_id = self.grammar.add_rule(rule)  # adds the rule to the grammar only if it's new

                if rule_id not in self.rule_id_to_record:
                    self.rule_id_to_record[rule_id] = Record(rule_id=rule_id)
                self.rule_id_to_record[rule_id].update(boundary_edges=boundary_edges, subtree=subtree, tnode=tnode)

                self.tnode_to_rule[tnode] = rule
            else:
                score = self.tnode_score(tnode=tnode, subtree=subtree)
                self.tnode_to_score[tnode] = score

            for kid in tnode.kids:
                if not kid.is_leaf:  # don't add to the bucket if it's a leaf
                    stack.append(kid)
        return

    def update_ancestor_scores(self, tnode: TreeNodeNew):
        """
        updates the scores of the ancestors
        :param tnode:
        :return:
        """
        active_nodes = set(self.g.nodes())
        tnode_leaves = tnode.leaves
        new_tnode_key = min(tnode_leaves)
        old_tnode_key = tnode.key
        tnode_children = tnode.children

        tnode = tnode.parent
        while tnode is not None:
            subtree = tnode.leaves & active_nodes
            tnode.leaves -= tnode_leaves
            tnode.leaves.add(new_tnode_key)  # tnode becomes a new leaf

            tnode.children.remove(old_tnode_key)   # remove the old tnode key from all subsequent ancestors
            tnode.children -= tnode_children
            tnode.children.add(new_tnode_key)

            self.tnode_to_score[tnode] = self.tnode_score(tnode=tnode, subtree=subtree)
            tnode = tnode.parent

    @abc.abstractmethod
    def update_tree(self, tnode: TreeNodeNew) -> None:
        """
        update the tree as needed - ancestors and descendants
        :param tnode:
        :return:
        """
        pass

    @abc.abstractmethod
    def tnode_score(self, tnode: TreeNodeNew, subtree: Set[int]) -> Any:
        """
        computes the score of a subtree
        :param subtree:
        :return:
        """
        pass

    @abc.abstractmethod
    def extract_rule(self) -> PartRule:
        """
        extracts one rule using the Extraction method
        :return:
        """
        pass

    def generate_grammar(self) -> None:
        """
        generates the grammar
        """
        start_time = time()
        num_nodes = self.g.order()

        is_global_extractor = hasattr(self, 'final_grammar')

        with tqdm(total=100, bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]', ncols=50) as pbar:
            while True:
                rule = self.extract_rule()
                assert rule is not None
                logging.debug(f'new rule: {rule}')

                if is_global_extractor:
                    self.final_grammar.add_rule(rule)
                else:
                    self.grammar.add_rule(rule)

                percent = (1 - (self.g.order() - 1) / (num_nodes - 1)) * 100
                pbar.update(percent - pbar.n)
                if rule.lhs == 0:  # we are compressing the root, so that's the end
                    break

        logging.warning(f'grammar generated in {round(time() - start_time, 3)} secs')
        if is_global_extractor:
            self.grammar = self.final_grammar


class MuExtractor(BaseExtractor):
    def __init__(self, g:LightMultiGraph, type: str, root: TreeNodeNew, grammar: VRG, mu: int):
        super().__init__(g=g, type=type, root=root, grammar=grammar, mu=mu)
        self.update_subtree_scores(start_tnode=self.root)  # initializes the scores

    def tnode_score(self, tnode: TreeNodeNew, subtree: Set[int]) -> Union[float, Tuple[float, int], Tuple[float, int, float]]:
        """
        returns infinity for rules > mu
        :param tnode:
        :param subtree:
        :return:
        """
        score = None
        diff = tnode.get_num_leaves() - self.mu
        if diff > 0:  # there are more nodes than mu
            mu_score = float('inf')
        elif diff < 0:
            mu_score = math.log2(1 - diff) # mu is greater
        else:
            mu_score = 0  # no penalty

        if self.type == 'mu_random':
            score =  mu_score # |mu - nleaf|
        elif self.type == 'mu_level':
            score = mu_score, tnode.level  # |mu - nleaf|, level of the tnode
        elif 'dl' in self.type:  # compute cost only if description length is used for scores
            if diff > 0: # don't bother creating the rule
                rule_cost = None
            else:
                rule, _ = create_rule(subtree=subtree, g=self.g, mode='part')
                rule.calculate_cost()
                rule_cost = rule.cost

            if self.type == 'mu_dl':
                score = mu_score, rule_cost
            elif self.type == 'mu_level_dl':
                score = mu_score, tnode.level, rule_cost

        assert score is not None, 'score is None'
        return score

    def update_tree(self, tnode: TreeNodeNew) -> None:
        """
        In this case, only update ancestors and their scores
        :param tnode:
        :return:
        """
        new_key = min(tnode.leaves)
        self.update_ancestor_scores(tnode=tnode)

        # delete score entries for all the subtrees
        del self.tnode_to_score[tnode]  # tnode now is a leaf
        for child in filter(lambda x: isinstance(x, str), tnode.children):
            del self.tnode_to_score[child]
        tnode.make_leaf(new_key=new_key)

    def extract_rule(self) -> PartRule:
        """
        Step 1: get best tnode
        Step 2: create rule, add to grammar
        Step 3: compress graph, update tree
        :return:
        """
        best_tnode, score = self.get_best_tnode_and_score()
        logging.debug(f'best tnode: {best_tnode}, score: {score}')
        subtree = best_tnode.leaves & set(self.g.nodes())

        rule, boundary_edges = create_rule(subtree=subtree, g=self.g, mode='part')

        compress_graph(boundary_edges=boundary_edges, g=self.g, subtree=subtree)
        self.update_tree(tnode=best_tnode)

        return rule


class LocalExtractor(BaseExtractor):
    """
    Uses local dl score to pick the best subtree. Treates each subtree as independent.
    """
    # __slots__ = ('__dict__', 'graph_dl', )

    def __init__(self, g:LightMultiGraph, type: str, root: TreeNodeNew, grammar: VRG, mu: int):
        super().__init__(g=g, type=type, root=root, grammar=grammar, mu=mu)
        self.graph_dl = graph_dl(self.g)  # during extraction, compute it once for all the rules because the graph doesn't change
        self.update_subtree_scores(start_tnode=self.root)

    def tnode_score(self, tnode: TreeNodeNew, subtree: Set[int]) -> None:
        """
        scores a tnode based on DL 
        for the extractor, lower scores are better - so we use the inverse of the score ,i.e.,
        score(tnode) = (dl(graph | rule) + dl(rule)) / dl(graph)
        :param tnode:
        :param subtree:
        :return:
        """
        if tnode.get_num_leaves() > self.mu:  # the rule size > mu
            score = float('inf')
        else:
            assert self.graph_dl is not None, 'Graph DL is not computed in tnode_score'
            g_copy = self.g.copy()  # make a copy

            rule, boundary_edges = create_rule(subtree=subtree, g=self.g, mode='part')
            rule.calculate_cost()
            rule_dl = rule.cost

            compress_graph(g=g_copy, subtree=subtree, boundary_edges=boundary_edges)  # compress the copy
            g_rule_dl = graph_dl(g_copy)  # after compression

            score = (g_rule_dl + rule_dl) / self.graph_dl
        return score

    def update_tree(self, tnode: TreeNodeNew) -> None:
        """
        In this case, only update ancestors and their scores
        :param tnode:
        :return:
        """
        new_key = min(tnode.leaves)
        self.update_ancestor_scores(tnode=tnode)

        # delete score entries for all the subtrees
        del self.tnode_to_score[tnode]  # tnode now is a leaf
        for child in filter(lambda x: isinstance(x, str), tnode.children):
            del self.tnode_to_score[child]
        tnode.make_leaf(new_key=new_key)

    def extract_rule(self) -> PartRule:
        """
        Step 0: compute graph dl
        Step 1: get best tnode
        Step 2: create rule, add to grammar
        Step 3: compress graph, update tree
        :return:
        """
        self.graph_dl = graph_dl(self.g)
        best_tnode, score = self.get_best_tnode_and_score()
        logging.debug(f'best tnode: {best_tnode}, score: {round(score, 3)}')
        subtree = best_tnode.leaves & set(self.g.nodes())

        rule, boundary_edges = create_rule(subtree=subtree, g=self.g, mode='part')

        compress_graph(boundary_edges=boundary_edges, g=self.g, subtree=subtree)
        self.update_tree(tnode=best_tnode)

        return rule


class GlobalExtractor(BaseExtractor):
    def __init__(self, g:LightMultiGraph, type: str, root: TreeNodeNew, grammar: VRG, mu: int):
        super().__init__(g=g, type=type, root=root, grammar=grammar, mu=mu)
        self.final_grammar = grammar.copy()
        self.graph_dl: float = graph_dl(self.g)  # during extraction, compute it once for all the rules because the graph doesn't change
        self.tnode_to_rule: Dict[TreeNodeNew, PartRule] = {}  # maps each tree node to a rule
        self.rule_id_to_record: Dict[int, Record] = {}  # maps each rule (via rule id) to a record object

        self.update_subtree_scores(start_tnode=self.root)
        self.init_record_scores()  # this updates the scores of the records
        logging.debug('Grammar initialized')

    def tnode_score(self, tnode: TreeNodeNew, subtree: Set[int]) -> Any:
        return None  # there is no need for tnode score in this case

    def update_ancestor_records(self, tnode: TreeNodeNew):
        """
        traverse up the tree, updating the rules and records
        :param tnode:
        :return:
        """
        active_nodes = set(self.g.nodes())
        tnode_leaves = tnode.leaves
        new_tnode_key = min(tnode_leaves)
        old_tnode_key = tnode.key
        tnode_children = tnode.children

        new_rule_id_to_record = dict(self.rule_id_to_record)
        tnode = tnode.parent

        g_copy = self.g.copy()
        g_dl = graph_dl(g_copy)

        while tnode is not None:
            tnode.leaves -= tnode_leaves
            tnode.leaves.add(new_tnode_key)  # tnode becomes a new leaf

            tnode.children.remove(old_tnode_key)  # remove the old tnode key from all subsequent ancestors
            tnode.children -= tnode_children
            tnode.children.add(new_tnode_key)
            subtree = tnode.leaves & active_nodes

            # we don't need to remove the boundary edges and subtrees
            old_rule = self.tnode_to_rule[tnode]
            old_rule.deactivate()

            old_record = self.rule_id_to_record[old_rule.id]
            old_record.remove()

            new_rule, boundary_edges = create_rule(subtree=subtree, g=self.g, mode='part')
            new_rule_id = self.grammar.add_rule(new_rule)  # add the new rule to the grammar

            if new_rule_id not in new_rule_id_to_record:
                new_rule_id_to_record[new_rule_id] = Record(rule_id=new_rule_id)
            new_rule_id_to_record[new_rule_id].update(boundary_edges=boundary_edges, subtree=subtree, tnode=tnode)
            self.set_record_score(record=new_rule_id_to_record[new_rule_id], g_copy=g_copy, g_dl=g_dl)

            self.tnode_to_rule[tnode] = new_rule
            # see if the rule exists elsewhere, otherwise disable
            # see if the new rule exists somewhere, then update the relevant record - shouldnt be too hard
            # the tricky part is to make sure that the grammar gets updated the way it should

            tnode = tnode.parent

        self.rule_id_to_record = new_rule_id_to_record


    def update_tree(self, tnode: TreeNodeNew) -> None:
        """
        update the tnode and the corresponding rule_id_to_record and tnode_to_rule data structures
        :param tnode:
        :return:
        """
        new_key = min(tnode.leaves)

        # update ancestors
        self.update_ancestor_records(tnode)

        # Disconnect the children
        rule = self.tnode_to_rule[tnode]
        rec = self.rule_id_to_record[rule.id]
        rec.remove()

        for child in filter(lambda x: isinstance(x, str), tnode.children):
            rule = self.tnode_to_rule[child]  # test this out
            rec = self.rule_id_to_record[rule.id]
            rec.remove()
        tnode.make_leaf(new_key=new_key)


        # clean up records
        empty_rec_rule_ids = [rec.rule_id for rec in self.rule_id_to_record.values() if rec.frequency == 0]
        for rule_id in empty_rec_rule_ids:
            del self.rule_id_to_record[rule_id]

        return

    def extract_rule(self) -> PartRule:
        """
        step 1: get best record
        step 2: for each tnode in the record
            step 2.1: update the tree rooted at the tree node
                step 2.1.1: the decendant rules get disabled only if they are not used elsewhere in the tree
                step 2.1.2: update the ancestors and their records and rules regardless
        :return:
        """
        best_record = self.get_best_record()
        best_rule = self.grammar[best_record.rule_id]

        # compress the graph & update the tree
        for subtree, boundary_edges, tnode in zip(best_record.subtree_list, best_record.boundary_edges_list,
                                                  best_record.tnodes_list):
            compress_graph(self.g, subtree=subtree, boundary_edges=boundary_edges)
            self.update_tree(tnode=tnode)

        return best_rule

    def get_best_record(self) -> Record:
        if len(self.rule_id_to_record) == 0:  # TODO fix BUG
            raise(Exception('Extraction Failed'))
        return min(self.rule_id_to_record.values(), key=lambda rec: rec.score)

    def set_record_score(self, record: Record, g_copy: LightMultiGraph, g_dl: float) -> Any:
        # compress the copied graph
        for tnode, subtree, boundary_edges in zip(record.tnodes_list, record.subtree_list, record.boundary_edges_list):
            compress_graph(g=g_copy, boundary_edges=boundary_edges, subtree=subtree)

        # compute the dl of the compressed graph
        g_rule_dl = graph_dl(g_copy)  # dl after compression

        # compute the score of the record
        rule = self.grammar[record.rule_id]
        rule.calculate_cost()
        rule_dl = rule.cost

        score = (rule_dl + g_rule_dl) / g_dl
        record.score = score
        return

    def init_record_scores(self):
        """
        updates the scores of the record objects -
        :return:
        """
        g_copy = self.g.copy()
        g_dl = graph_dl(self.g)  # initial graph dl

        for record in self.rule_id_to_record.values():
            rule = self.grammar[record.rule_id]

            # the rule is larger than mu
            if rule.graph.order() > self.mu:
                record.score = float('inf')
                continue

            assert rule.frequency == record.frequency, 'the frequencies of the rule and record should match'

            self.set_record_score(record, g_copy=g_copy, g_dl=g_dl)
        return

