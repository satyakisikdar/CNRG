import networkx as nx
import vrgs.MDL as MDL

class BaseRule:
    """
    Base class for Rule
    """
    def __init__(self):
        self.lhs = 0  # the left hand side: the number of boundary edges
        self.graph = nx.MultiGraph()  # the right hand side subgraph
        self.level = set()  # level of discovery in the tree (the root is at 0)
        self.cost = 0  # the cost of encoding the rule using MDL (in bits)
        self.frequency = 1  # frequency of occurence

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



class FullRule(BaseRule):
    """
    Rule object for full-info option
    """
    def __init__(self):
        super().__init__()
        self.internal_nodes = set()  # the set of internal nodes
        self.edges_covered = set()  # edges in the original graph that's covered by the rule

    def calculate_cost(self):
        """
        Updates the MDL cost of the RHS. l_u is the number of unique entities in the graph.
        We have two types of nodes (internal and external) and one type of edge
        :return:
        """
        self.cost = len(MDL.gamma_code(self.lhs + 1)) + MDL.graph_mdl_v2(self.graph, l_u=3) + \
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
        iso_nodes = set()
        for node in self.graph.nodes_iter():
            if isinstance(node, int) and self.graph.degree(node) == 1:  # identifying the isolated nodes
                iso_nodes.add(node)

        if len(iso_nodes) == 0:  # the rule cannot be contracted
            return

        rhs_copy = self.graph.copy()

        [self.graph.add_edge(u, 'I', attr_dict={'b': True})  # add the new edges
         for iso_node in iso_nodes
         for u in rhs_copy.neighbors_iter(iso_node)]

        self.graph.remove_nodes_from(iso_nodes)   # remove the old isolated nodes



class PartRule(BaseRule):
    """
    Rule class for Partial option
    """
    def __init__(self):
        super().__init__()

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
        b_deg = nx.get_edge_attributes(self.graph, 'b_deg')
        max_boundary_degree = max(b_deg.values())

        self.cost = len(MDL.gamma_code(self.lhs + 1)) + MDL.graph_mdl_v2(self.graph, l_u=2) + \
                    len(MDL.gamma_code(self.frequency + 1)) + len(MDL.gamma_code(max_boundary_degree + 1))


class NoRule(PartRule):
    """
    Class for no_info
    """
    def calculate_cost(self):
        """
        Calculates the MDL for the rule. This just includes encoding the graph.
        l_u = 2 (because we have one type of nodes and one type of edge)
        :return:
        """
        self.cost = len(MDL.gamma_code(self.lhs + 1)) + MDL.graph_mdl_v2(self.graph, l_u=2) + \
                    len(MDL.gamma_code(self.frequency + 1))