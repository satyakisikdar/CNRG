import networkx as nx
import vrgs.MDL as MDL
import re

class Rule:
    def __init__(self, ):
        self.lhs = 0  # the left hand side: the number of boundary edges
        self.rhs = nx.MultiGraph()  # the right hand side subgraph
        self.level = set()  # level of discovery in the tree (the root is at 0)
        self.edges_covered = set()  # edges in the original graph that's covered by the rule
        self.cost = 0  # the cost of encoding the rule using MDL (in bits)
        self.internal_nodes = set()  # the set of internal nodes
        self.frequency = 1  # number of times this rule appears

    def __str__(self):
        st = '{} -> (n = {}, m = {})'.format(self.lhs, self.rhs.order(), self.rhs.size())
        if self.frequency > 1:  # if freq > 1, show it in square brackets
            st += '[{}]'.format(self.frequency)
        return st

    def __repr__(self):
        st = '{} -> ({}, {})'.format(self.lhs, self.rhs.order(), self.rhs.size())
        if self.frequency > 1:  # if freq > 1, show it in square brackets
            st += '[{}]'.format(self.frequency)
        return st

    def __eq__(self, other):  # two rules are equal if the LHSs match and RHSs are isomorphic
        g1 = nx.convert_node_labels_to_integers(self.rhs)
        g2 = nx.convert_node_labels_to_integers(other.rhs)
        return self.lhs == other.lhs \
                and nx.is_isomorphic(g1, g2)

    # def __hash__(self):  # two rules hash to the same location if they share the same LHS
    #     # rhs = self.rhs.copy()
    #     # nx.freeze(rhs)
    #     return hash((self.lhs))#, rhs))

    def calculate_cost(self):
        """
        Updates the MDL cost of the RHS
        :return:
        """
        self.cost = len(MDL.gamma_code(self.lhs + 1)) + MDL.graph_mdl_v2(self.rhs)

    def generalize_rhs(self):
        """
        Relabels the RHS such that the internal nodes are Latin characters, the boundary nodes are numerals.
        TODO - is this general enough? Can we make it easier to merge.

        :param self: RHS subgraph
        :return:
        """
        mapping = {}
        internal_node_counter = 'a'
        boundary_node_counter = 0

        for n in self.internal_nodes:
            mapping[n] = internal_node_counter
            internal_node_counter = chr(ord(internal_node_counter) + 1)

        for n in [x for x in self.rhs.nodes_iter() if x not in self.internal_nodes]:
            mapping[n] = boundary_node_counter
            boundary_node_counter += 1
        self.rhs = nx.relabel_nodes(self.rhs, mapping=mapping)

    def contract_rhs(self):
        """
        Contracts the RHS such that all boundary nodes with degree 1 are replaced by a special boundary isolated node I
        """
        iso_nodes = set()
        for node in self.rhs.nodes_iter():
            if isinstance(node, int) and self.rhs.degree(node) == 1:  # identifying the isolated nodes
                iso_nodes.add(node)

        if len(iso_nodes) == 0:  # the rule cannot be contracted
            return

        rhs_copy = self.rhs.copy()

        [self.rhs.add_edge(u, 'I', attr_dict={'b': True})  # add the new edges
         for iso_node in iso_nodes
         for u in rhs_copy.neighbors_iter(iso_node)]

        self.rhs.remove_nodes_from(iso_nodes)   # remove the old isolated nodes
