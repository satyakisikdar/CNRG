### VRG updates
05/31
- **Relevant papers**
  - <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.220.2503&rep=rep1&type=pdf"> OPAvion: Mining and Visualization in Large Graphs </a>
    - Pegasus has a large scale eigensolver to find *iteresting* patterns 
  - <a href="http://eda.mmci.uni-saarland.de/pubs/2015/vog-koutra,kang,vreeken,faloutsos-2015-sam.pdf">VOG: Summarizing and Understanding Large Graphs</a>
    - Summarizes graphs - given a graph, finds set of overlapping subgraphs, to most succintly describe the given graph
    - Uses MDL to come up with a quality function - a collection $M$ of structures has a description length $L(G, M)$, which is the quality score. 
    - Use MDL to identify the structure type of the candidates
    - VoG gives a list of overlapping subgraphs sorted in importance - structures that save the most bits, i.e. achieve best compression
    - Minimizes $L(M) + L(D | M)$, $L(M)$ - length in bits of description of M, $L(D|M)$ - length in bits of the description of the data when encoded with $M$. Called two-part crude MDL, because it uses both graph and the model. Vocabulary $\Omega=\{fc, nc, fb, nb, ch, st\}$ f, n, c, b, ch, st = full, near, clique, bipartite core, chain, star.  
    - Each structure $s \in M$ identifies a patch of the adj matrix $A$, known as $area(s, M, A)$ i.e., $(i, j) \in A$ where $i$ and $j$ are in $s$. 
    - Nodes can overlap, edges are counted only in the first structure that they appear in.
    - 
    
  - Reducing large graphs to small supergraphs: a unified approach
  - <a href="https://people.csail.mit.edu/jshun/6886-s18/papers/Liu2018.pdf">Graph Summarization: A Survey</a>
  - <a href="http://reports-archive.adm.cs.cmu.edu/anon/anon/usr/ftp/2015/CMU-CS-15-126.pdf">Exploring and Making Sense of Large Graphs</a>
  - <a href="http://www.vldb.org/pvldb/vol8/p1924-koutra.pdf">Perseus: an interactive large-scale graph mining and visualization tool</a>
  - <a href="http://vialab.science.uoit.ca/wp-content/papercite-data/pdf/ver2017.pdf">Optimizing hierarchical visualizations with the minimum description length principle</a>
- **Possible experiments**
  - Planting cliques and other well known structures and adding variable noise to test recoverability 
- Stretch out current method with graphs - real and with planted partitions 
- Read Danai Koutra's and related works to figure out experiments to do 
- Run METIS, and wrap it in the code. 
-------
05/24
- SUBDUE does not scale at all. Try MPI. Ask Tim to install MPI Daemon on dsg2

05/23 
- Top-k substructures, finite number of iterations of subdue, ...
- Print the bags of nodes in SUB_x and OVERLAP_y.
- Update on reporting all instances of a substructure. Overlap works only if there are multiple iterations. Set flag ```-iterations 0```. 
- Test SUBDUE and Motif Cluster on medium to large graphs.  

05/21 
- SUBDUE (C) does not report all instances of subgraph. Recursive, overlap flags didn't help.
- Trying out the Python version... Writing a wrapper for generating edgelists as a JSON. 

05/19
- Parser for subdue output works. 
- Need to feed it to the Julia / Matlab code

05/17
- Find motifs using SUBDUE using the occurrences and find conductance of the motifs using Benson
- Check if they match  
- Check if greedy does get you the optimum clusters 
- Nature of conductance sweep plot - is it convex? *NO* http://snap.stanford.edu/higher-order/high_order-netsci-may16a.pdf slide #29. 
- best substructure, top-k substructures - does that make a difference? 
