# make the other metrics work
# generate the txt files, then work on the pdf otuput
__version__ = "0.1.0"
import networkx as nx
import numpy as np
import pandas as pd
from glob import glob
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import matplotlib.pylab as pylab
params = {'legend.fontsize':'small',
          'figure.figsize': (1.6 * 10, 1.0 * 8),
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize':'small',
          'ytick.labelsize':'small'}
pylab.rcParams.update(params)
import matplotlib.gridspec as gridspec

def plot_timestamped_graphs(list_of_nx_graphs,pos=None,outfigname="tmp.pdf"):
    gs = gridspec.GridSpec(1, len(list_of_nx_graphs))
    for k in range(len(list_of_nx_graphs)):
        axh = plt.subplot(gs[0,k])
        g = list_of_nx_graphs[k]
        if pos is None:
            nx.draw_networkx(g, nx.spring_layout(g))
        else:
            nx.draw_networkx(g, pos=pos)
        plt.tick_params(axis='both', left='off', top='off',right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        axh.spines['right'].set_visible(False)
        axh.spines['top'].set_visible(False)
        axh.spines['left'].set_visible(False)

    plt.savefig(outfigname,bbox_inches='tight')
