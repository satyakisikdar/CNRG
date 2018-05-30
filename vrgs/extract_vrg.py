import networkx as nx

g = nx.MultiDiGraph()
g.add_edge(1,3)
g.add_edge(2,1)
g.add_edge(2,5)
g.add_edge(3,4)
g.add_edge(4,5)
g.add_edge(4,2)
g.add_edge(4,9)
g.add_edge(5,1)
g.add_edge(5,3)

g.add_edge(6,2)
g.add_edge(6,7)
g.add_edge(6,8)
g.add_edge(6,9)
g.add_edge(7,8)
g.add_edge(9,8)
g.add_edge(9,6)

def find_boundary_edges(sg, g):
    # collect all of the boundary edges (i.e., the edges
    # that connect the subgraph to the original graph)
    boundary_edges = list()
    for n in sg:
        if g.is_directed():
            boundary_edges += g.in_edges(n)
            boundary_edges += g.out_edges(n)
        else:
            boundary_edges += g.edges(n)

    # remove internal edges from list of boundary_edges
    boundary_edges = [x for x in boundary_edges if x not in sg.edges()]

    return boundary_edges


def generalize_rhs(sg):
    # TODO remove the original graph's labels from the RHS subgraph
    return sg


def extract_vrg(g, tree):
    if not isinstance(tree, list):
        # if we are at a leaf, then we need to backup one level
        return
    for index, subtree in enumerate(tree):
        # build the grammar from a left-to-right bottom-up tree ordering (in order traversal)
        extract_vrg(g, subtree)
        if not isinstance(subtree, list):
            # if we are at a leaf, then we need to backup one level
            continue

        # subtree to be replaced
        print(subtree)

        sg = g.subgraph(subtree)
        print(sg.edges())
        boundary_edges = find_boundary_edges(sg, g)
        for u, v in boundary_edges:
            sg.add_edge(u, v, attr_dict={'b': True})

        lhs = len(boundary_edges)

        # next we contract the original graph
        [g.remove_node(n) for n in subtree]
        new_node = min(subtree)
        g.add_node(new_node, attr_dict={'label': lhs})

        # rewire new_node
        for u, v in boundary_edges:
            if u in subtree:
                u = new_node
            if v in subtree:
                v = new_node
            g.add_edge(u,v)

        rhs = generalize_rhs(sg)
        print(g.nodes(data=True))

        rhs = generalize_rhs(sg)

        #replace subtree with new_node
        tree[index] = new_node


tree = [[[1,2], [[3,4], 5]], [[9,8], [6,7]]]

extract_vrg(g, tree)
