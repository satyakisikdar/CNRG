import networkx as nx
import random as r

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
    in_edges = list()
    out_edges = list()
    for n in sg:
        if g.is_directed():
            in_edges += g.in_edges(n)
            out_edges += g.out_edges(n)
        else:
            in_edges += g.edges(n)

    # remove internal edges from list of boundary_edges
    in_edges = [x for x in in_edges if x not in sg.edges()]
    out_edges = [x for x in out_edges if x not in sg.edges()]

    return in_edges, out_edges


def generalize_rhs(sg, internal_nodes):
    # remove the original graph's labels from the RHS subgraph
    nodes = {}
    internal_node_counter = 'a'
    boundary_node_counter = 0
    rhs = nx.MultiDiGraph()

    for n in internal_nodes:
        rhs.add_node(internal_node_counter, sg.node[n])
        nodes[n] = internal_node_counter
        internal_node_counter = chr(ord(internal_node_counter) + 1)
    for n in [x for x in sg.nodes() if x not in internal_nodes]:
        rhs.add_node(boundary_node_counter, sg.node[n])
        nodes[n] = boundary_node_counter
        boundary_node_counter += 1
    for u, v, d in sg.edges(data=True):
        rhs.add_edge(nodes[u], nodes[v], attr_dict=d)
    return rhs


def extract_vrg(g, tree):
    vrg = list()
    if not isinstance(tree, list):
        # if we are at a leaf, then we need to backup one level
        return vrg
    for index, subtree in enumerate(tree):
        # build the grammar from a left-to-right bottom-up tree ordering (in order traversal)
        vrg += extract_vrg(g, subtree)
        if not isinstance(subtree, list):
            # if we are at a leaf, then we need to backup one level
            continue

        # subtree to be replaced
        print(subtree)

        sg = g.subgraph(subtree)
        print(sg.edges())
        boundary_edges = find_boundary_edges(sg, g)
        for direction in range(0, len(boundary_edges)):
            for u, v in boundary_edges[direction]:
                sg.add_edge(u, v, attr_dict={'b': True})

        lhs = (len(boundary_edges[0]), len(boundary_edges[1]))

        # next we contract the original graph
        [g.remove_node(n) for n in subtree]
        new_node = min(subtree)
        g.add_node(new_node, attr_dict={'label': lhs})

        # rewire new_node
        for direction in range(0, len(boundary_edges)):
            for u, v in boundary_edges[direction]:
                if u in subtree:
                    u = new_node
                if v in subtree:
                    v = new_node
                g.add_edge(u,v)

        rhs = generalize_rhs(sg, subtree)
        print(g.nodes(data=True))

        #replace subtree with new_node
        tree[index] = new_node
        vrg += [(lhs, rhs)]
    return vrg


def stochastic_vrg(vrg):
    # Create a new graph from the VRG at random

    node_counter = 1
    non_terminals = set()
    new_g = nx.MultiDiGraph()

    new_g.add_node(0, attr_dict={'label': (0, 0)})
    non_terminals.add(0)

    while len(non_terminals) > 0:
        # continue until no more non-terminal nodes

        # choose a non terminal node at random
        node_sample = r.sample(non_terminals, 1)[0]
        lhs = new_g.node[node_sample]['label']
        print('Selected node ' + str(node_sample) + ' with label ' + str(lhs))

        rhs = r.sample(vrg[lhs], 1)[0]

        singleton = nx.MultiDiGraph()
        singleton.add_node(node_sample)
        broken_edges = find_boundary_edges(singleton, new_g)
        assert(len(broken_edges[0]) == lhs[0] and len(broken_edges[1]) == lhs[1])

        new_g.remove_node(node_sample)
        non_terminals.remove(node_sample)

        nodes = {}

        for n, d in rhs.nodes(data=True):
            if isinstance(n, str):
                new_node = node_counter
                nodes[n] = new_node
                new_g.add_node(new_node, attr_dict=d)
                if 'label' in d:
                    non_terminals.add(new_node)
                node_counter += 1

        # randomly assign broken edges to boundary edges
        r.shuffle(broken_edges[0])
        r.shuffle(broken_edges[1])

        # wire the broken edge
        for u, v, d in rhs.edges(data=True):
            if 'b' in d:
                # boundry edge
                if isinstance(u, str):
                    # outedges
                    choice = r.sample(broken_edges[1],1)[0]
                    new_g.add_edge(nodes[u], choice[1])
                    broken_edges[1].remove(choice)
                else:
                    # inedges
                    choice = r.sample(broken_edges[0], 1)[0]
                    new_g.add_edge(choice[0], nodes[v])
                    broken_edges[0].remove(choice)
            else:
                # internal edge
                new_g.add_edge(nodes[u], nodes[v])
    return new_g


tree = [[[[1,2], [[3,4], 5]], [[9,8], [6,7]]]]
vrg = extract_vrg(g, tree)

vrg_dict = {}
# we need to turn the list into a dict for efficient access to the LHSs
for lhs, rhs in vrg:
    if lhs not in vrg_dict:
        vrg_dict[lhs] = [rhs]
    else:
        vrg_dict[lhs].append(rhs)

new_g = stochastic_vrg(vrg_dict)
print(len(new_g.edges()))
