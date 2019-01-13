Code repository for the paper *Modelling graphs with Vertex Replacement Grammars* by 
*Satyaki Sikdar, Justus Hibshman, Tim Weninger*. Submitted to ECML-PKDD 2019 journal track.
All graph generation stats can be found in `all_stats.csv`. 

Run `python3 -m pip install -r requirements.txt` prior to running `runner.py` to install required packages.

Usage instructions (also visible by running `python3 runner.py -h`)
```
usage: runner.py [-h] [-g] [-c] [-b {full,part,no}] [-l LAMB] [-s] [-o OUTDIR]
                 [-n N]

optional arguments:
  -h, --help            show this help message and exit
  -g , --graph          Name of the graph (default: karate)
  -c , --clustering     Clustering method to use (default: louvain)
  -b {full,part,no}, --boundary {full,part,no}
                        Degree of boundary information to store (default: part)
  -l LAMB, --lamb LAMB  Size of RHS (lambda) (default: 5)
  -s , --selection      Selection strategy (default: level)
  -o OUTDIR, --outdir OUTDIR
                        Name of the output directory (default: output)
  -n N                  Number of graphs to generate (default: 5)
```
