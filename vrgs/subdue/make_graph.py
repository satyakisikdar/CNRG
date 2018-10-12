import networkx as nx

g = nx.karate_club_graph()

f = open('./karate.g', 'w')

for u in g.nodes_iter():
  f.write(f'\nv {u+1} v')

for u, v in g.edges_iter():
  f.write(f'\ne {u+1} {v+1} e')

f.close()

