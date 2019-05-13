clusters = {}  # stores the cluster members
original_graph = None   # to keep track of the original edges covered

def find_boundary_edges(g, nbunch):
    """
    Collect all of the boundary edges (i.e., the edges
    that connect the subgraph to the original graph)

    :param g: whole graph
    :param nbunch: set of nodes in the subgraph
    :return: boundary edges
    """
    nbunch = set(nbunch)
    if len(nbunch) == g.order():  # it's the entire node set
        return []
    boundary_edges = []
    for u in nbunch:
        for v in g.neighbors(u):
            if v not in nbunch:
                edges = [(u, v)] * g.number_of_edges(u, v)
                boundary_edges.extend(edges)
    return boundary_edges

