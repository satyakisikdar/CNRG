from time import sleep

import networkx as nx
from tqdm import tqdm


class LightMultiGraph(nx.Graph):
    def __init__(self):
        nx.Graph.__init__(self)

    def size(self, weight=None):
        return int(super(LightMultiGraph, self).size(weight='weight'))

    def __repr__(self):
        return f'n = {self.order():_d} m = {self.size():_d}'

    def add_edge(self, u, v, attr_dict=None, **attr):
        # print(f'inside add_edge {u}, {v}')
        if attr_dict is not None and 'weight' in attr_dict:
            wt = attr_dict['weight']
        elif attr is not None and 'weight' in attr:
            wt = attr['weight']
        else:
            wt = 1
        if self.has_edge(u, v):  # edge already exists
            # print(f'edge ({u}, {v}) exists, {self[u][v]["weight"]}')
            self[u][v]['weight'] += wt
        else:
            super(LightMultiGraph, self).add_edge(u, v, weight=wt)

    def copy(self):
        g_copy = LightMultiGraph()
        for node, d in self.nodes(data=True):
            if len(d) == 0:  # prevents adding an empty 'attr_dict' dictionary
                g_copy.add_node(node)
            else:
                if 'label' in d:  # this keeps the label and the b_deg attributes to the same level
                    g_copy.add_node(node, label=d['label'])

        for e in self.edges(data=True):
            u, v, d = e
            g_copy.add_edge(u, v, attr_dict=d)
        return g_copy

    def add_edges_from(self, ebunch, attr_dict=None, **attr):
        for e in ebunch:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = {}  # doesnt need edge_attr_dict_factory
            else:
                raise nx.NetworkXError(
                    "Edge tuple %s must be a 2-tuple or 3-tuple." % (e,))
            self.add_edge(u, v, attr_dict=dd, **attr)

    def number_of_edges(self, u=None, v=None):
        if u is None:
            return self.size()
        try:
            return self[u][v]['weight']
        except KeyError:
            return 0  # no such edge


if __name__ == '__main__':
    # g = RHSGraph()
    # g.add_edge(1, 2)
    # g.add_edge(2, 3)
    # g.add_edge(1, 2)
    #
    # print(g.edges(data=True))
    # print(g.number_of_edges(1, 2))
    with tqdm(total=100) as pbar:
        perc = 0
        while perc <= 100:
            sleep(0.1)
            # pbar.update(pbar.n - perc)
            pbar.update(perc - pbar.n)
            perc += 5