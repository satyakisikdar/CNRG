G = dlmread('./data/karate.mat');
G = sparse(G);
graphname = 'karate';

nnodes = size(G, 1);
nedges = nnz(G) / 2;
fprintf('nodes: %d edges: %d\n', nnodes, nedges);

nd = accumarray(nonzeros(sum(G,2)),1);
maxdegree = find(nd>0,1,'last');
fprintf('Maximum degree: %d\n', maxdegree);

[ccd,gcc] = ccperdeg(G);
fprintf('Global clustering coefficient: %.2f\n', gcc);

fprintf('Running BTER...\n');
t1=tic;
[E1,E2] = bter(nd,ccd);
toc(t1)
fprintf('Number of edges created by BTER: %d\n', size(E1,1) + size(E2,1));

fprintf('Turning edge list into adjacency matrix (including dedup)...\n');
t2=tic;
G_bter = bter_edges2graph(E1,E2);
toc(t2);
fprintf('Number of edges in dedup''d graph: %d\n', nnz(G)/2);

% Number of nodes and edges
nnodes_bter = size(G_bter,1);
nedges_bter = nnz(G_bter)/2;
fprintf('Comparative Results: %s versus BTER\n', graphname);
fprintf('Number of nodes: %d versus %d\n', nnodes, nnodes_bter);
fprintf('Number of edges: %d versus %d\n', nedges, nedges_bter);

G_bter = full(G_bter);
dlmwrite('karate_bter.mat', G_bter, ' ');
