"""
VRG extraction
"""
import abc
import logging
from typing import List, Tuple, Dict, DefaultDict, Set, Any, Union, Optional
import math
from time import time
import networkx as nx
from tqdm import tqdm
import itertools
import pickle 

from src.LightMultiGraph import LightMultiGraph
from src.MDL import graph_dl
from src.Rule import FullRule, NoRule, PartRule
from src.globals import find_boundary_edges
from src.part_info import set_boundary_degrees
from src.Tree import TreeNode
from src.VRG import VRG

class Record:
    __slots__ = 'tnodes_list', 'rule_id', 'frequency', 'boundary_edges_list', 'subtree_list', 'score'

    def __init__(self, rule_id: int):
        self.tnodes_list: List[TreeNode] = []
        self.rule_id: int = rule_id
        self.frequency: int = 0  # number of times we have seen this rule_id
        self.boundary_edges_list: List[Set[Tuple[int, int]]] = []
        self.subtree_list: List[Set[int]] = []
        self.score = None  # score of the rule

    def update(self, boundary_edges: Any, subtree: Set[int], tnode: TreeNode):
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


def compress_graph(g: LightMultiGraph, subtree: Set[int], boundary_edges: Any, permanent: bool) -> Union[None, float]:
    """
    :param g: the graph
    :param subtree: the set of nodes that's compressed
    :param boundary_edges: boundary edges
    :param permanent: if disabled, undo the compression after computing the new dl -> returns the float
    :return:
    """
    assert len(subtree) > 0, f'Empty subtree g:{g.order(), g.size()}, bound: {boundary_edges}'
    before = (g.order(), g.size())

    if not isinstance(subtree, set):
        subtree = set(subtree)

    if boundary_edges is None:
        # compute the boundary edges
        boundary_edges = find_boundary_edges(g, subtree)

    removed_edges = set()
    removed_nodes = set()
    # step 1: remove the nodes from subtree, keep track of the removed edges
    if not permanent:
        removed_edges = list(g.subgraph(subtree).edges(data=True))
        removed_nodes = list(g.subgraph(subtree).nodes(data=True))
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

    if not permanent:  # if this flag is set, then return the graph dl of the compressed graph and undo the changes
        compressed_graph_dl = graph_dl(g)
        # print(f'In compress_graph, dl after change: {compressed_graph_dl:_g}')
        g.remove_node(new_node)  # and the boundary edges
        g.add_nodes_from(removed_nodes)  # add the subtree

        for e in itertools.chain(removed_edges, boundary_edges):
            if len(e) == 3:
                u, v, d = e
            else:
                u, v = e
                d = {'weight': 1}
            g.add_edge(u, v, weight=d['weight'])

        after = (g.order(), g.size())
        assert before == after, 'Decompression did not work'
        return compressed_graph_dl
    else:
        return None


class BaseExtractor(abc.ABC):
    # __slots__ = 'type', 'g', 'root', 'tnode_to_score', 'grammar', 'mu'

    def __init__(self, g:LightMultiGraph, type: str, root: TreeNode, grammar: VRG, mu: int) -> None:
        assert type in ('local_dl', 'global_dl', 'mu_random', 'mu_level', 'mu_dl', 'mu_level_dl'), f'Invalid mode: {type}'
        self.type = type
        self.g = g  # the graph
        self.root = root
        self.tnode_to_score: Dict[TreeNode, Any] = {}
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

    def update_subtree_scores(self, start_tnode: TreeNode) -> Any:
        """
        updates scores of the tnodes of the subtree rooted at start_tnode depending on the extraction type
        :param start_tnode: starting tnode. for the entire tree, use self.root
        :return:
        """
        active_nodes = set(self.g.nodes())
        stack: List[TreeNode] = [start_tnode]
        nodes_visited = 0
        total_tree_nodes = len([child for child in start_tnode.children if isinstance(child, str)]) + 1 # +1 for root
        is_global_extractor = hasattr(self, 'rule_id_to_record')

        # logging.warning('Updaing the tree')
        # with tqdm(total=100, bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]', ncols=50) as pbar:
        while len(stack) != 0:
            tnode = stack.pop()
            nodes_visited += 1
            subtree = tnode.leaves & active_nodes

            if is_global_extractor:  # this is true for GlobalExtractor objects only
                self.rule_id_to_record: Dict[int, Record]

                rule, boundary_edges = create_rule(subtree=subtree, g=self.g, mode='part')
                rule_id = self.grammar.add_rule(rule)  # adds the rule to the grammar only if it's new, updates otherwise

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
                # perc = (nodes_visited / total_tree_nodes) * 100
                # progress = perc - pbar.n
                # pbar.update(progress)
        return

    def update_ancestor_scores(self, tnode: TreeNode):
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
        is_global_extractor = hasattr(self, 'rule_id_to_record')

        tnode = tnode.parent
        while tnode is not None:
            subtree = tnode.leaves & active_nodes
            tnode.leaves -= tnode_leaves
            tnode.leaves.add(new_tnode_key)  # tnode becomes a new leaf

            tnode.children.discard(old_tnode_key)   # remove the old tnode key from all subsequent ancestors  # switched from remove to discard
            tnode.children -= tnode_children
            tnode.children.add(new_tnode_key)
            if not is_global_extractor:
                self.tnode_to_score[tnode] = self.tnode_score(tnode=tnode, subtree=subtree)
            tnode = tnode.parent
        return

    @abc.abstractmethod
    def update_tree(self, tnode: TreeNode) -> None:
        """
        update the tree as needed - ancestors and descendants
        :param tnode:
        :return:
        """
        pass

    @abc.abstractmethod
    def tnode_score(self, tnode: TreeNode, subtree: Set[int]) -> Any:
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
        # tqdm.write(f'Extracting grammar name:{self.grammar.name} mu:{self.grammar.mu} type:{self.grammar.type} clustering:{self.grammar.clustering}')
        with tqdm(total=100, bar_format='{l_bar}{bar}|[{elapsed}<{remaining}]', ncols=50) as pbar:
            while True:
                rule = self.extract_rule()
                assert nx.is_connected(self.g), 'graph is disonnected'
                assert rule is not None
                logging.debug(f'new rule: {rule}')

                if is_global_extractor:
                    self.final_grammar.add_rule(rule)
                else:
                    self.grammar.add_rule(rule)

                percent = (1 - (self.g.order() - 1) / (num_nodes - 1)) * 100
                curr_progress = percent - pbar.n
                pbar.update(curr_progress)
                if rule.lhs == 0:  # we are compressing the root, so that's the end
                    assert self.g.order() == 1, 'Graph not correctly compressed'
                    break

        if is_global_extractor:
            self.grammar = self.final_grammar
        logging.warning(f'{self.grammar} generated in {round(time() - start_time, 3)} secs\n')


class MuExtractor(BaseExtractor):
    def __init__(self, g:LightMultiGraph, type: str, root: TreeNode, grammar: VRG, mu: int):
        super().__init__(g=g, type=type, root=root, grammar=grammar, mu=mu)
        self.update_subtree_scores(start_tnode=self.root)  # initializes the scores

    def tnode_score(self, tnode: TreeNode, subtree: Set[int]) -> Union[float, Tuple[float, int], Tuple[float, int, float]]:
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

    def update_tree(self, tnode: TreeNode) -> None:
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
        logging.debug(f'\nbest tnode: {best_tnode}, score: {score}')
        subtree = best_tnode.leaves & set(self.g.nodes())

        rule, boundary_edges = create_rule(subtree=subtree, g=self.g, mode='part')

        compress_graph(g=self.g, subtree=subtree, boundary_edges=boundary_edges, permanent=True)
        self.update_tree(tnode=best_tnode)

        return rule


class LocalExtractor(BaseExtractor):
    """
    Uses local dl score to pick the best subtree. Treates each subtree as independent.
    """
    # __slots__ = ('__dict__', 'graph_dl', )

    def __init__(self, g:LightMultiGraph, type: str, root: TreeNode, grammar: VRG, mu: int):
        super().__init__(g=g, type=type, root=root, grammar=grammar, mu=mu)
        self.graph_dl = graph_dl(self.g)  # during extraction, compute it once for all the rules because the graph doesn't change
        self.update_subtree_scores(start_tnode=self.root)

    def tnode_score(self, tnode: TreeNode, subtree: Set[int]) -> None:
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

            rule, boundary_edges = create_rule(subtree=subtree, g=self.g, mode='part')
            rule.calculate_cost()
            rule_dl = rule.cost

            g_rule_dl = compress_graph(g=self.g, subtree=subtree, boundary_edges=boundary_edges, permanent=False)  # compress the copy
            assert g_rule_dl is not None, 'compress graph returns None'
            score = (g_rule_dl + rule_dl) / self.graph_dl
        return score

    def update_tree(self, tnode: TreeNode) -> None:
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

        compress_graph(g=self.g, subtree=subtree, boundary_edges=boundary_edges, permanent=True)
        self.update_tree(tnode=best_tnode)

        return rule


class GlobalExtractor(BaseExtractor):
    def __init__(self, g:LightMultiGraph, type: str, root: TreeNode, grammar: VRG, mu: int):
        super().__init__(g=g, type=type, root=root, grammar=grammar, mu=mu)
        self.final_grammar = grammar.copy()
        self.graph_dl: float = graph_dl(
            self.g)  # during extraction, compute it once for all the rules because the graph doesn't change
        self.tnode_to_rule: Dict[TreeNode, PartRule] = {}  # maps each tree node to a rule
        self.rule_id_to_record: Dict[int, Record] = {}  # maps each rule (via rule id) to a record object

        self.update_subtree_scores(start_tnode=self.root)
        self.update_all_record_scores()  # this updates the scores of the records
        logging.debug('Grammar initialized')

    def tnode_score(self, tnode: TreeNode, subtree: Set[int]) -> Any:
        return None  # there is no need for tnode score in this case

    def get_best_record(self) -> Record:
        assert len(self.rule_id_to_record) > 0, 'Empty records, extraction failed'
        return min(self.rule_id_to_record.values(), key=lambda rec: rec.score)

    def update_tree(self, tnode: TreeNode) -> None:
        """
        update the tnode and the corresponding rule_id_to_record and tnode_to_rule data structures
        :param tnode:
        :return:
        """
        new_key = min(tnode.leaves)
        self.update_ancestor_scores(tnode)
        tnode.make_leaf(new_key=new_key)
        return

    def extract_rule(self) -> PartRule:
        """
        step 1: get best record
        step 2: for each tnode in the record
            step 2.1: update the tree rooted at the tree node
                step 2.1.1: the decendant rules get disabled only if they are not used elsewhere in the tree
                step 2.1.2: update the ancestors and their records and rules regardless
        step 3: update ALL the record scores after extraction since the graph changes.
        :return:
        """
        # step 1: get best record
        best_record = self.get_best_record()
        best_rule = self.grammar[best_record.rule_id]

        # tqdm.write(f'{best_record.tnodes_list}, {best_record.subtree_list}, {best_rule}')

        # step 2: compress graph, then update tree
        for tnode, subtree, boundary_edges in zip(best_record.tnodes_list, best_record.subtree_list, best_record.boundary_edges_list):
            subtree = set(subtree) & set(self.g.nodes())  # take only the subtree things that are in the graph
            compress_graph(g=self.g, boundary_edges=None, subtree=subtree, permanent=True)
            self.update_tree(tnode=tnode)

        # step 3: update all the data structures
        if best_rule.lhs == 0:
            assert self.g.order() == 1, 'Improper extraction, since the graph has > 1 nodes'
        else:
            # reset the data structures
            self.rule_id_to_record = {}
            self.tnode_to_rule = {}
            self.grammar.reset()

            self.update_subtree_scores(start_tnode=self.root)  # update all the subtree
            self.update_all_record_scores()
        return best_rule

    def set_record_score(self, record: Record, g_dl: float) -> None:
        # compress the copied graph
        g_rule_dl = 0
        for tnode, subtree, boundary_edges in zip(record.tnodes_list, record.subtree_list, record.boundary_edges_list):
            # try:
            assert len(subtree) > 0, 'empty subtree'
            g_rule_dl = compress_graph(g=self.g, subtree=subtree, boundary_edges=boundary_edges, permanent=False)

        # compute the score of the record
        rule = self.grammar[record.rule_id]
        rule.calculate_cost()
        rule_dl = rule.cost

        score = (rule_dl + g_rule_dl) / g_dl
        record.score = score
        return

    def update_all_record_scores(self) -> None:
        """
        updates the scores of all the record objects -
        :return:
        """
        g_dl = graph_dl(self.g)  # initial graph dl

        for record in self.rule_id_to_record.values():
            rule = self.grammar[record.rule_id]

            # the rule is larger than mu
            if rule.graph.order() > self.mu:
                record.score = float('inf')
                continue

            assert rule.frequency == record.frequency, 'the frequencies of the rule and record should match'
            self.set_record_score(record, g_dl=g_dl)
        return

if __name__ == '__main__':
    name = 'lesmis'
    outdir = 'output'
    # clustering = 'leiden'
    clustering = 'cond'
    type = 'mu_level'
    mu = 3

    g_ = nx.Graph()
    g_.add_edges_from([(1, 2), (1, 3), (1, 5),
                      (2, 4), (2, 5),
                      (3, 4), (3, 5), (4, 5),
                      (2, 7), (4, 9),
                      (6, 7), (6, 8), (6, 9),
                      (7, 8), (7, 9), (8, 9)])
    g = LightMultiGraph()
    g.add_edges_from(g_.edges())
    root = pickle.load(open('../output/trees/sample/cond_tree.pkl', 'rb'))
    print(root)

    grammar = VRG(clustering=clustering, type=type, name=name, mu=mu)

    # extractor = MuExtractor(g=g, type=type, mu=mu, grammar=grammar, root=root)
    # extractor = LocalExtractor(g=g, type=type, mu=mu, grammar=grammar, root=root)
    extractor = GlobalExtractor(g=g, type=type, mu=mu, grammar=grammar, root=root)

    key2node = {}
    s = [extractor.root]
    while len(s) != 0:
        tnode = s.pop()
        key2node[tnode.key] = tnode
        for kid in tnode.kids:
            if not kid.is_leaf:
                s.append(kid)

    extractor.update_tree(tnode=key2node['f'])
    print()
