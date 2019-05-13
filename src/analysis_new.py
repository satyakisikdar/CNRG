"""
New analysis script

1. degree dist
2. hop-plot
3. avg clustering coeff by degree
4. graphlet counts and gcd
5. also netcomp
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

from src.GCD import GCD
from src.LightMultiGraph import LightMultiGraph


def compute_hop_plot(g):
    """
    computes the hop-plot of g
    :param g:
    :return:
    """

def plot_degree_dist(degree_dist):
    plt.xlabel('Degree')
    plt.ylabel('Fraction of Nodes')
    plt.plot(degree_dist, marker='o', alpha=0.5)
    plt.show()

class Analyzer:
    def __init__(self, original_graph: nx.Graph, generated_graphs: List[LightMultiGraph]):
        self.original_graph = original_graph
        self.generated_graphs = generated_graphs
        self.gcds: List[float] = []  # gcds of the generated graphs

    def compute_GCDs(self):
        for gen_graph in self.generated_graphs:
            gcd = GCD(self.original_graph, gen_graph, mode='orca')
            self.gcds.append(gcd)

    def compute_mean_degree_dist(self):
        max_n = max(g.order() for g in self.generated_graphs)
        avg_degree_dist = np.zeros(max_n)

        for gen_graph in self.generated_graphs:
            deg_dist =  np.array(nx.degree_histogram(g))
            deg_dist = deg_dist / gen_graph.order()  # normalize
            avg_degree_dist = avg_degree_dist + deg_dist

        avg_degree_dist = avg_degree_dist / max_n
        return avg_degree_dist

if __name__ == '__main__':
    g = nx.barabasi_albert_graph(100, 2, seed=1)
    compute_degree_dist(g)