import networkx as nx
import vrgs.MDL as MDL

class BaseRule:
    """
    Base class for Rule
    """
    def __init__(self, lhs=0, graph=nx.MultiGraph(), level=0, cost=0, frequency=1):
        self.lhs = lhs  # the left hand side: the number of boundary edges
        self.graph = graph  # the right hand side subgraph
        self.level = level  # level of discovery in the tree (the root is at 0)
        self.cost = cost  # the cost of encoding the rule using MDL (in bits)
        self.frequency = frequency  # frequency of occurence

    def __str__(self):
        st = '{} -> (n = {}, m = {})'.format(self.lhs, self.graph.order(), self.graph.size())
        if self.frequency > 1:  # if freq > 1, show it in square brackets
            st += '[{}]'.format(self.frequency)
        return st

    def __repr__(self):
        st = '{} -> ({}, {})'.format(self.lhs, self.graph.order(), self.graph.size())
        if self.frequency > 1:  # if freq > 1, show it in square brackets
            st += '[{}]'.format(self.frequency)
        return st

    def __eq__(self, other):  # two rules are equal if the LHSs match and RHSs are isomorphic
        g1 = nx.convert_node_labels_to_integers(self.graph)
        g2 = nx.convert_node_labels_to_integers(other.graph)
        return self.lhs == other.lhs \
                and nx.is_isomorphic(g1, g2)

    def __hash__(self):
        g = nx.freeze(self.graph)
        return hash((self.lhs, g))

    # def __del__(self):
    #     print('del rule')

    def __deepcopy__(self, memodict={}):
        return BaseRule(lhs=self.lhs, graph=self.graph, level=self.level, cost=self.cost, frequency=self.frequency)

    def contract_rhs(self):
        pass


class FullRule(BaseRule):
    """
    Rule object for full-info option
    """
    def __init__(self, lhs=0, graph=nx.MultiGraph(), level=0, cost=0, frequency=1, internal_nodes=set(),
                 edges_covered=set()):
        super().__init__(lhs=lhs, graph=graph, level=level, cost=cost, frequency=frequency)
        self.internal_nodes = internal_nodes  # the set of internal nodes
        self.edges_covered = edges_covered  # edges in the original graph that's covered by the rule

    def __deepcopy__(self, memodict={}):
        return FullRule(lhs=self.lhs, graph=self.graph, level=self.level, cost=self.cost, frequency=self.frequency,
                        internal_nodes=self.internal_nodes, edges_covered=self.edges_covered)

    def calculate_cost(self):
        """
        Updates the MDL cost of the RHS. l_u is the number of unique entities in the graph.
        We have two types of nodes (internal and external) and one type of edge
        :return:
        """
        self.cost = len(MDL.gamma_code(self.lhs + 1)) + MDL.graph_mdl(self.graph, l_u=3) + \
                    len(MDL.gamma_code(self.frequency + 1))

    def generalize_rhs(self):
        """
        Relabels the RHS such that the internal nodes are Latin characters, the boundary nodes are numerals.

        :param self: RHS subgraph
        :return:
        """
        mapping = {}
        internal_node_counter = 'a'
        boundary_node_counter = 0


        for n in self.internal_nodes:
            mapping[n] = internal_node_counter
            internal_node_counter = chr(ord(internal_node_counter) + 1)

        for n in [x for x in self.graph.nodes_iter() if x not in self.internal_nodes]:
            mapping[n] = boundary_node_counter
            boundary_node_counter += 1
        self.graph = nx.relabel_nodes(self.graph, mapping=mapping)
        self.internal_nodes = {mapping[n] for n in self.internal_nodes}


    def contract_rhs(self):
        """
        Contracts the RHS such that all boundary nodes with degree 1 are replaced by a special boundary isolated node I
        """
        # iso_nodes = set()
        # for node in self.graph.nodes_iter():
        #     if isinstance(node, int) and self.graph.degree(node) == 1:  # identifying the isolated nodes
        #         iso_nodes.add(node)
        #
        # if len(iso_nodes) == 0:  # the rule cannot be contracted
        #     return
        #
        # rhs_copy = self.graph.copy()
        #
        # [self.graph.add_edge(u, 'I', attr_dict={'b': True})  # add the new edges
        #  for iso_node in iso_nodes
        #  for u in rhs_copy.neighbors_iter(iso_node)]
        #
        # self.graph.remove_nodes_from(iso_nodes)   # remove the old isolated nodes
        return


class PartRule(BaseRule):
    """
    Rule class for Partial option
    """
    def __init__(self, lhs=0, graph=nx.MultiGraph(), level=0, cost=0, frequency=1):
        super().__init__(lhs=lhs, graph=graph, level=level, cost=cost, frequency=frequency)

    def __deepcopy__(self, memodict={}):
        return PartRule(lhs=self.lhs, graph=self.graph, level=self.level, cost=self.cost, frequency=self.frequency)

    def generalize_rhs(self):
        """
        Relabels the RHS such that the internal nodes are Latin characters, the boundary nodes are numerals.

        :param self: RHS subgraph
        :return:
        """
        mapping = {}
        internal_node_counter = 'a'

        for n in self.graph.nodes_iter():
            mapping[n] = internal_node_counter
            internal_node_counter = chr(ord(internal_node_counter) + 1)

        self.graph = nx.relabel_nodes(self.graph, mapping=mapping)

    def calculate_cost(self):
        """
        Calculates the MDL for the rule. This includes the encoding of boundary degrees of the nodes.
        l_u = 2 (because we have one type of nodes and one type of edge)
        :return:
        """
        b_deg = nx.get_node_attributes(self.graph, 'b_deg')
        max_boundary_degree = max(b_deg.values())

        self.cost = len(MDL.gamma_code(self.lhs + 1)) + MDL.graph_mdl(self.graph, l_u=2) + \
                    len(MDL.gamma_code(self.frequency + 1)) + len(MDL.gamma_code(max_boundary_degree + 1))


class NoRule(PartRule):
    """
    Class for no_info
    """
    def __deepcopy__(self, memodict={}):
        return NoRule(lhs=self.lhs, graph=self.graph, level=self.level, cost=self.cost, frequency=self.frequency)

    def calculate_cost(self):
        """
        Calculates the MDL for the rule. This just includes encoding the graph.
        l_u = 2 (because we have one type of nodes and one type of edge)
        :return:
        """
        self.cost = len(MDL.gamma_code(self.lhs + 1)) + MDL.graph_mdl(self.graph, l_u=2) + \
                    len(MDL.gamma_code(self.frequency + 1))