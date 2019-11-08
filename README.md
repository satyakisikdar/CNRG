Code repository for the paper *Modelling graphs with Vertex Replacement Grammars* by 
*Satyaki Sikdar, Justus Hibshman, and Tim Weninger*. Find the preprint <a href="https://arxiv.org/abs/1908.03837">here</a>.

Run `python3 -m pip install -r requirements.txt` prior to running `runner.py` to install required packages.

Usage instructions (also visible by running `python3 runner.py -h`)
```
usage: runner.py [-h] [-g] [-c] [-b {full,part,no}] [-l LAMB] [-s] [-o OUTDIR]
                 [-n N]

optional arguments:
  -h, --help            Show this help message and exit
  -g, --graph           Name of the graph, looks in the ./src/tmp directory for the edge list (default: karate)
  -c, --clustering      Clustering method to use (default: louvain)
  -b {full,part,no}, --boundary {full,part,no}
                        Degree of boundary information to store (default: part)
  -l , --lamb LAMB      Size of RHS (lambda) (default: 5)
  -s , --selection      Selection strategy (default: level)
  -o , --outdir OUTDIR  Name of the output directory (default: output)
  -n N                  Number of graphs to generate (default: 5)
```

Edge lists of new undirected graphs can be added in the `./src/temp/` directory with extension `.g`.
