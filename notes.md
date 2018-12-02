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
    - $M$ encodes a part of the matrix $A$, defined by the $area$ of substructures in $M$. So, there is an error $E = M \xor A$. Knowing $M$ and $E$, you can regenerate $A$ losslessly. $L(G, M) = L(M) + L(E)$. 
    - First step - find subgraphs - either thru Subdue, METIS, or other community detection techniques.
    - Length of a model $M$ is defined based on #structures and the encoding length per structure. Each clique, bipartite core, star, chain is encoded. 
    - *GreedyNForget* - consider each structure in $C$ in descending order of quality, as long as the total encoded cost of the model does not increase, include the structure, otherwise reject.
    - Tested with Cavemen graphs (n=841, m=~7,500)- two cliques separated by stars
    
  - <a href="http://web.eecs.umich.edu/~dkoutra/papers/18_Condense-SNAM.pdf">Reducing large graphs to small supergraphs: a unified approach</a>
    - Summarizes the structure of a given  network by selecting a small set of its most informative structural patterns 
    - Searches local structures that optimize MDL - can deal with overlaps - using pre-defined structural patterns - cliques, stars, bipartite cores, chains, *hyperbolic structures* with skewed distributions. 
    - (a) Uses community detection, (b) then MDL to minimize edge redundancy by the structures, and (c) iterative, divide and conquer policy for reducing selection bias of substructures.
    - $L(G, M) = L(M) + L(E) + L(O)$, $E$ and $O$ are the error and overlap matrices.
    - Hyperbolic structures - skewed degree dist, with powerlaw exponent between -0.6 and -1.5.  
    - They set no of clusters for METIS and SPECTRAL to $\sqrt{\frac{n}{2}}$ as a rule of thumb
    - Metrics used - (a) Conciseness - compression rates, (b) Minimal redundancy - minimal overlap of supernodes, and (c) Coverage - number of nodes and edges covered by the summaries
    
  - <a href="https://people.csail.mit.edu/jshun/6886-s18/papers/Liu2018.pdf">Graph Summarization: A Survey</a>
    
  - <a href="http://www.cs.umd.edu/hcil/trs/2012-29/2012-29.pdf">Motif Simplication: Improving Network Visualization</a>
    - Uses 3 motifs - fans, connectors, and cliques  
  - <a href="http://reports-archive.adm.cs.cmu.edu/anon/anon/usr/ftp/2015/CMU-CS-15-126.pdf">Exploring and Making Sense of Large Graphs</a>
  - <a href="http://www.vldb.org/pvldb/vol8/p1924-koutra.pdf">Perseus: an interactive large-scale graph mining and visualization tool</a>
  - <a href="http://vialab.science.uoit.ca/wp-content/papercite-data/pdf/ver2017.pdf">Optimizing hierarchical visualizations with the minimum description length principle</a>
- **Possible experiments**
  - Planting cliques and other well known structures and adding variable noise to test recoverability 
  - Test out VRG on Cavemen graphs 
  - Stretch out current method with graphs - real and with planted partitions 
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
