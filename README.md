Notes

### VRG updates
05/31
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
