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
    boundary_edges = []
    for u, v in g.edges_iter():
        if u in nbunch and v not in nbunch:
            boundary_edges.append((u, v))
        elif u not in nbunch and v in nbunch:
            boundary_edges.append((u, v))
    return boundary_edges