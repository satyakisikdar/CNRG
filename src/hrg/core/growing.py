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
          'figure.figsize': (8, 4),
          'axes.labelsize': 'small',
          'axes.titlesize': 'small',
          'xtick.labelsize':'small',
          'ytick.labelsize':'small'}
pylab.rcParams.update(params)
import matplotlib.gridspec as gridspec
import matplotlib.patches  as mpatches
import math
import sys
import os
import PHRG
import probabilistic_cfg as pcfg
import net_metrics as metrics
import load_edgelist_from_dataframe as tdf
import argparse,traceback,optparse
import pprint as pp
import plot_timestamped_graphs as ptsg




def Hstar_Graphs_Ignore_Time(df, graph_name, tslices, axs):
  if len(df.columns) == 3:
    G = nx.from_pandas_dataframe(df, 'src', 'trg', edge_attr='ts')
  else:
    G = nx.from_pandas_dataframe(df, 'src', 'trg')
  # force to unrepeated edgesA
  if 0: print nx.info(G)
  G = G.to_undirected()
  if 0: print nx.info(G)
  exit()
  # Derive the prod rules in a naive way, where
  prod_rules = PHRG.probabilistic_hrg_learning(G)
  g = pcfg.Grammar('S')
  for (id, lhs, rhs, prob) in prod_rules:
    g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

  num_nodes = G.number_of_nodes()

  print "Starting max size"
  g.set_max_size(num_nodes)

  print "Done with max size"

  Hstars = []

  num_samples = 20
  print '*' * 40
  for i in range(0, num_samples):
    rule_list = g.sample(num_nodes)
    hstar = PHRG.grow(rule_list, g)[0]
    Hstars.append(hstar)

  # if 0:
  #   g = nx.from_pandas_dataframe(df, 'src', 'trg', edge_attr=['ts'])
  #   draw_degree_whole_graph(g,axs)
  #   draw_degree(Hstars, axs=axs, col='r')
  #   #axs.set_title('Rules derived by ignoring time')
  #   axs.set_ylabel('Frequency')
  #   axs.set_xlabel('degree')

  if 1:
    # metricx = [ 'degree','hops', 'clust', 'assort', 'kcore','eigen','gcd']
    metricx = ['eigen']
    g = nx.from_pandas_dataframe(df, 'src', 'trg',edge_attr=['ts'])
    # graph_name = os.path.basename(f_path).rstrip('.tel')
    print ">", graph_name
    metrics.network_properties( [g], metricx, Hstars, name=graph_name, out_tsv=True)


def pandas_dataframes_from_edgelists(el_files):
  if (el_files is None):  return
  list_of_dataframes = []
  for f in el_files:
    print '~'*80
    print f
    temporal_graph = False
    with open(f,'r') as ifile:
      line = ifile.readline()
      while (not temporal_graph):
        if ("%" in line):
          line = ifile.readline()
        elif len(line.split()) > 3 :
          temporal_graph = True
    if (temporal_graph):
      dat = np.genfromtxt(f, dtype=np.int64, comments='%', delimiter="\t", usecols=[0, 1, 3], autostrip=True)
      df = pd.DataFrame(dat, columns=['src', 'trg', 'ts'])
    else:
      dat = np.genfromtxt(f, dtype=np.int64, comments='%', delimiter="\t", usecols=[0, 1], autostrip=True)
      df = pd.DataFrame(dat, columns=['src', 'trg'])
    df = df.drop_duplicates()
    list_of_dataframes.append(df)

  return list_of_dataframes

def Structure_Varying_Overtime(df, hrBlck, axs):
  # import datetime
  # red_patch = mpatches.Patch(color='red', label='uniq nodes')
  # blu_patch = mpatches.Patch(color='blue', label='edges')
  print '{} hr'.format(hrBlck)
  dat = {}
  clqs = {}
  agg_hrs = 0
  for s in range(df['ts'].min(), df['ts'].max(),int(3600*hrBlck)):
    mask = (df['ts'] >= s) & (df['ts'] < s+ 3600*hrBlck)
    tdf = df.loc[mask]
    agg_hrs +=hrBlck
    SG = nx.from_pandas_dataframe(tdf, 'src', 'trg', ['ts'])
    dat[agg_hrs] = np.mean(SG.degree().values())
    cliq=nx.find_cliques(SG)
    clqs[agg_hrs] = np.mean([len(c) for c in cliq])
  xvals = sorted(dat.keys())
  #print [datetime.datetime.fromtimestamp(d).strftime("%d/%m") for d in xvals]
  yvals = [dat[x] for x in xvals]
  axs.plot(xvals, yvals,'.',linestyle="-",label='Avg degree')
  # Save to disk the need files
  with open ("Results/avg_degree_structure_in_{}hrs.tsv".format(hrBlck),'w') as f:
    for k in range(0,len(yvals)):
      f.write("({},{})\n".format(xvals[k],yvals[k]))

  yvals = [clqs[x] for x in xvals]
  axs.plot(xvals, yvals,'.', linestyle="-", label="Avg clique size")
  axs.set_xlabel('hours')
  # Save to disk the need files
  with open ("Results/avg_cliq_size_structure_in_{}hrs.tsv".format(hrBlck),'w') as f:
    for k in range(0,len(yvals)):
      f.write("({},{})\n".format(xvals[k],yvals[k]))
  return



def plot_node_accumulation(df,axs):
  red_patch = mpatches.Patch(color='red', label='uniq nodes')
  blu_patch = mpatches.Patch(color='blue', label='edges')

  gb = df.groupby(['ts'])

  accum_nodes = [(k,len(set(df.loc[v].src).union(set(df.loc[v].trg)))) for k,v in gb.groups.items()]
  df = pd.DataFrame(accum_nodes)
  df.columns = ['ts','uniq_nodes']
  df.plot.scatter(x=['ts'], y=['uniq_nodes'],color='b', ax=axs, alpha=0.5)

  # # axs0.set_title(r'Baseline (ignore time, train and predict on the complete graph)')
  # # axs0.set_xscale('log')     #open(results_folder+'.done/'+g_gname,'a').close()
  # axs.set_ylabel('count')
  # axs.set_xlabel('time (epochs)')
  # plt.legend(handles=[red_patch, blu_patch])

def plot_number_of_edges(df,axs):
  red_patch = mpatches.Patch(color='red', label='edges')
  blu_patch = mpatches.Patch(color='blue', label='nodes')

  print 'processing dataframe'
  gb = df.groupby(['ts'])

  if 0:
    nodes_per_ts = [nx.from_pandas_dataframe(df.loc[v], 'src', 'trg', ['ts']).number_of_nodes() for k,v in gb.groups.items()]
    axs.plot(gb.groups.keys(),nodes_per_ts,'.r', alpha=0.5,label="nodes") # number of nodes per `ts`
    #
    edges_per_ts = [nx.from_pandas_dataframe(df.loc[v], 'src', 'trg', ['ts']).number_of_edges() for k,v in gb.groups.items()]
    axs.plot(gb.groups.keys(),edges_per_ts,'.b', alpha=0.5,label="edges") # number of nodes per `ts`

  # Using pd.DatFrame to count # of edges
  gb = gb.count()
  gb['w'].plot(ax=axs,marker='x',linestyle="", alpha=0.75)

  # axs0.set_title(r'Baseline (ignore time, train and predict on the complete graph)')
  # axs0.set_xscale('log')     #open(results_folder+'.done/'+g_gname,'a').close()
  axs.set_ylabel('count')
  axs.set_xlabel('time (epochs)')
  axs.legend(handles=[red_patch, blu_patch])

def plot_kthslice_number_of_edges(pddf,axhandle,kSlice,nSlices):
  '''
  Plot only the kth slice to the given axhandle
  Args:
    pddf:
    axhandle:
    tslice:

  Returns:

  '''
  # span = float(tslice/nSlices)
  span = (pddf['ts'].max() - pddf['ts'].min())/nSlices
  mask = (pddf['ts'] >= pddf['ts'].min()+ span*kSlice) & (pddf['ts'] < pddf['ts'].min()+ span*(kSlice +1))
  pddf = pddf.loc[mask]
  gb = pddf.groupby(['ts']).count()
  gb['w'].plot(ax=axhandle,marker='.',linestyle="", alpha=0.75)

def Growing_Network_Nodes_Edges(pddf, nSlices, kSlice, axs):
  '''
  Evolving Node Count Plot
  Args:
     pddf: pandas dataframe
  nSlices: total # blocks
   kSlice: current slice
      axs: axes handle to plot

  Returns:

  '''

  print 'Nodes & edges accum. in block:',kSlice+1
  span = (pddf['ts'].max() - pddf['ts'].min())/nSlices
  mask = (pddf['ts'] >= pddf['ts'].min()+ span*kSlice) & (pddf['ts'] < pddf['ts'].min()+ span*(kSlice +1))
  pddf = pddf.loc[mask]
  gb   = pddf.groupby(['ts'])
  accum_nodes = [(k,len(set(pddf.loc[v].src).union(set(pddf.loc[v].trg)))) for k,v in gb.groups.items()]
  df = pd.DataFrame(accum_nodes)
  accum_sum=0
  b = []
  for i in df.index:
    if i == 0: accum_sum = df[1].iloc[i]; b.append([df[0].iloc[i], accum_sum])
    else: accum_sum += df[1].iloc[i]; b.append([df[0].iloc[i],accum_sum])
  df = pd.DataFrame(b)
  df[1].plot(x=[0],ax=axs, label='Accum Nodes')
  axs.set_ylabel('Evolving Node Count')
  axs.set_xlabel('epochs for block {}'.format(kSlice))
  return


def Growing_Network_Using_Final_State_ProdRules(pddf, prod_rules, nSlices, kSlice, axs):
  '''
  Grow a synthetic graph up to the end of block kSlice using HRG rules
  from the final (whole) state of the graph.
        pddf: pandas df
  prod_rules: production rules learned on the entire graph
     nSlices: total number of blocks (pseudo-states of the graph)
      kSlice: the current slice
         axs: axes to plot to
  '''

  span = (pddf['ts'].max() - pddf['ts'].min())/nSlices

  g = pcfg.Grammar('S')
  for (id, lhs, rhs, prob) in prod_rules:
    g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

  # mask = (pddf['ts'] >= pddf['ts'].min()+ span*kSlice) & (pddf['ts'] < pddf['ts'].min()+ span*(kSlice +1))
  mask = (pddf['ts'] >= pddf['ts'].min()) & (pddf['ts'] < pddf['ts'].min()+ span*(kSlice +1))
  ldf = pddf.loc[mask]

  G = nx.from_pandas_dataframe(ldf, 'src', 'trg', ['ts'])

  num_nodes = G.number_of_nodes()
  print "Starting max size"
  g.set_max_size(num_nodes)
  print "Done with max size"

  num_samples = 20
  print '*' * 40
  tdf = pd.DataFrame()
  for i in range(0, num_samples):
    rule_list = g.sample(num_nodes)
    hstar = PHRG.grow(rule_list, g)[0]
    df  = pd.DataFrame.from_dict(hstar.degree().items())
    #
    tdf = pd.concat([df.groupby([1]).count(),df.groupby([1]).count()], axis=1)

  tdf = tdf[0].mean(axis=1)
  tdf.plot(ax=axs,color='r', label='Orig')
  # Orig Graph
  tdf = pd.DataFrame.from_dict(G.degree().items())
  gb  = tdf.groupby([1]).count()
  gb[0].plot(ax=axs, color='b', label='Orig')
  axs.set_xscale('log')

  '''
  TODO CONTINUE THIS AND KEEP WORKING ON GETTING THE DEGREE PLOTTED
  '''

def plot_nbr_prod_rules_per_ts(pddf, axs, kthSlice, nSlices):
  span = (pddf['ts'].max() - pddf['ts'].min())/nSlices
  mask = (pddf['ts'] >= pddf['ts'].min()+ span*kthSlice) & (pddf['ts'] < pddf['ts'].min()+ span*(kthSlice +1))
  print pddf.shape
  pddf = pddf.loc[mask]
  print pddf.shape

  # sg = nx.from_pandas_dataframe(pddf, 'src', 'trg', ['ts'])
  # if 0: print nx.info(sg)
  # cliq=nx.find_cliques(sg)
  # print sorted((len(c) for c in cliq))

  gb = pddf.groupby(['ts']).groups
  ts_cliq_cnt = {}
  for k in gb.keys():
    df = pddf.loc[gb[k]]
    sg = nx.from_pandas_dataframe(df, 'src', 'trg', ['ts'])
    cliq=nx.find_cliques(sg)
    ts_cliq_cnt[k] = [len(c) for c in cliq]

  df = pd.DataFrame.from_dict(ts_cliq_cnt.items(),dtype=np.int64)
  df['av'] = df[1].apply(lambda x: np.mean(x))

  df.sort_values(by=[0],inplace=True)
  # print df.head()
  # df['av'].plot(x=[0],ax=axs,color='b',alpha=0.75)
  axs.plot(df[0].values, df['av'].values,'b', alpha=0.6)
  axs.set_xlabel('epochs')
  axs.set_ylabel('Avg Clique Length')

  return # [_]

def find_smallest_timestamp_delta(df, gname, nSlices=None):

    # ABS smallest delta
    gb = df.groupby(['ts']).groups
    nf = pd.DataFrame(gb.keys())
    nf.sort_values(by=[0],inplace=True)

    nf['delta']=nf.diff()
    # print nf.head()
    print gname, 'absolute min t_delta', nf.delta.min(), 'units'
    span = pd.to_datetime(df['ts'].max(),unit='s') -pd.to_datetime(df['ts'].min(), unit='s')
    # span = df['ts'].max() - df['ts'].min()
    print gname, 'span:', span


    # blck = (df['ts'].max() - df['ts'].min())/float(nSlices)
    # #____________________hrs   days   weeek   months  years
    # timeblocks = [1,60, 3600, 86400, 604800, 2629743,31556926 ]
    #
    # tWin = span/float(nSlices)
    # min_delta_perblock =[]
    # for slc in range(0,nSlices):
    #   mask = (df['ts'] >= df['ts'].min()+tWin*slc) & (df['ts'] <= df['ts'].min()+tWin*(slc+1))
    #   ldf = df.loc[mask]
    #   gb = ldf.groupby(['ts']).groups
    #   nf = pd.DataFrame(gb.keys())
    #   if len(nf)==0:
    #     min_delta_perblock.append(0)
    #     continue
    #   nf.sort_values(by=[0],inplace=True)
    #   nf['delta'] = nf.diff()
    #   min_delta_perblock.append(nf.delta.min())
    #
    # print 'The min delta per slice:', min_delta_perblock
    # # print [nf.delta.min(), np.max(min_delta_perblock)]
    # time_blocks = []
    # for tblck in timeblocks:
    #   if tblck > np.max([nf.delta.min(), np.max(min_delta_perblock)]):
    #     #time_blocks.append(tblck*60)
    #     time_blocks.append(timeblocks[timeblocks.index(tblck)+1])
    #     time_blocks.append(timeblocks[timeblocks.index(tblck)])
    #     break

    return  nf.delta.min()


def get_abs_path(net_name_str):
  print net_name_str
  net_name_str = net_name_str.strip('\'')
  from subprocess import Popen, PIPE
  # p = call(["find", ".", "-iname","out.{}*".format(net_name_str)])
  p = Popen(['find', "../PhoenixPython", "-iname", "out.{}*filtered".format(net_name_str), '-type', 'f'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
  re_stdout = p.stdout.read()
  if re_stdout == "":
    p = Popen(['find', "../PhoenixPython", "-iname", "out.{}*".format(net_name_str), '-type', 'f'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    re_stdout = p.stdout.read()
  return re_stdout

def accumulation_of_nodes(pddf, netname, out2pdf=False):
  gb = pddf.groupby(['ts'])
  net_evo_bynodes = {}
  for k in gb.groups.keys():
    # print k
    # print gb.groups[k]
    # print pddf.loc[gb.groups[k]]
    net_evo_bynodes[k]= len(set(pddf.loc[gb.groups[k]].src).union(set(pddf.loc[gb.groups[k]].trg)))

  nf = pd.DataFrame.from_dict(net_evo_bynodes.items())
  nf['xnorm'] = nf.index.values/float(len(nf))
  nf['accum'] = nf[0].cumsum()
  nf['anorm'] = nf['accum']/nf.accum.max()

  gs = gridspec.GridSpec(2, 1)
  ax0 =  plt.subplot(gs[0, 0])
  ax1 =  plt.subplot(gs[1, 0])

  nf.plot.scatter(x=['xnorm'],y=[1],facecolors='none', edgecolors='b', ax=ax0,alpha=0.5, label='$n$ per uniq t.s.')
  nf.plot(x=['xnorm'],y=['anorm'],color='r',ax=ax1,label='Accumulation of nodes')
  nf[['xnorm','anorm']].to_csv('/tmp/n_accum_{}.tsv'.format(netname),sep='\t',header=False, index=None)

  if out2pdf:
    outfigname = '/tmp/outfig_{}.pdf'.format(netname)
    plt.savefig(outfigname, bbox_inches='tight')
    if os.path.exists(outfigname): print 'Output: {}'.format(outfigname)
  return

def min_time_info(pd_df, graph_name):
  # if graph_name_arg is None:
  #   graph_name_arg = "Contact"
  ''' Find the smallest timestamp delta '''
  print '-'*20,'Find the smallest timestamp delta','-'*20
  if g_name == 'slashdot':
    pd_df = pd_df[pd_df['ts']>0]
  slc_in_secs = find_smallest_timestamp_delta(pd_df)

  k = 0
  unix_epoch = {3600: 'hrs', 86400: 'days', 604800: 'weeks', 2629743: 'months', 31556926: 'years'}
  print 'Plottting...', slc_in_secs

def evolving_mean_degree(pd_df,pd_gbg):
  '''
  evolving average degree
  Args:
    pd_df:
    pd_gbg:

  Returns:

  '''
  meank = {}
  for k in pd_gbg.keys():
    df = pd_df.loc[pd_gbg[k]]
    g  = nx.from_pandas_dataframe(df, 'src', 'trg', ['ts'])
    df = pd.DataFrame.from_dict(g.degree().items())
    gb = df.groupby([1]).count()
    meank[k] = gb[0].mean()

  return meank

def simpleaxis(ax):
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_left()
  # ax.get_xaxis().set_ticks([])
  ax.set_xticklabels([])
  ax.grid(True)

def plot_evolving_structure_inblocks(pd_df, tslices, graph_name):
  if g_name == 'slashdot':
    pd_df = pd_df[pd_df['ts']>0]

  slice = (pd_df.ts.max() - pd_df.ts.min())/tslices
  block_mean_degree = []
  for blk in range(pd_df.ts.min(),pd_df.ts.max(),slice):
    print blk, pd_df.ts.min(), pd_df.ts.max()
    mask = (pd_df['ts'] >= blk) & (pd_df['ts'] <= blk+slice)
    ldf = pd_df.loc[mask]
    if not ldf.shape[0]:
      block_mean_degree.append(0)
      continue
    G = nx.from_pandas_dataframe(ldf, 'src', 'trg', ['ts'])
    df = pd.DataFrame.from_dict(G.degree().items())
    gb = df.groupby([1]).count()
    block_mean_degree.append(gb[0].mean())

    break

  gs  = gridspec.GridSpec(1, 1)
  ax3 = plt.subplot(gs[0, 0])
  ax3.plot(range(0,len(block_mean_degree)), block_mean_degree, '.',linestyle="-", alpha=0.5, label=r'$\langle k \rangle$')
  h,l=ax3.get_legend_handles_labels() # get labels and handles from ax1
  ax3.legend(h,l)
  ax3.spines['top'].set_visible(False)
  ax3.spines['right'].set_visible(False)
  ax3.set_ylabel(r'Mean degree $\langle k \rangle$')
  ax3.grid(True)
  with open ('/tmp/{}_mean_degree_in{}tblocks.tsv'.format(graph_name,tslices),mode='w') as f:
    [f.write("{}\t{}\n".format(x,block_mean_degree[x])) for x in range(0,len(block_mean_degree))]

def plot_evolving_structure(pd_df, graph_name):
  if g_name == 'slashdot':
    pd_df = pd_df[pd_df['ts']>0]
  print 'pd_df',pd_df.shape

  span=(pd_df['ts'].max() - pd_df['ts'].min())
  gs  = gridspec.GridSpec(3, 1)
  ax3 = plt.subplot(gs[2, 0])
  ax1 = plt.subplot(gs[0, 0])
  ax2 = plt.subplot(gs[1, 0])
  ax3.set_xlabel("time")
  # edges_dict = {}

  # mask = (pd_df['ts'] >= pd_df['ts'].min()) & (pd_df['ts'] <= pd_df['ts'].min()+ 1000)
  # pd_df = pd_df.loc[mask]
  pd_df = pd_df.head(1000)
  print 'pd_df',pd_df.shape

  gb = pd_df.groupby(['ts']).groups
  df = pd.DataFrame(gb.keys(),columns=['k'])
  df['m'] = [len(x) for x in gb.values()]
  df['n'] = [nx.from_pandas_dataframe(pd_df.loc[x],'src', 'trg', ['ts']).number_of_nodes()
                    for x in gb.values()]
  df['cs_m'] = df['m'].cumsum()
  df['cs_n'] = df['n'].cumsum()
  avgk_d = evolving_mean_degree(pd_df,gb)

  ax1.plot(df.index,df['m'],color='b',marker='.', alpha=0.75 ,label="edges (m)")
  ax1.plot(df.index,df['n'],color='r',marker='.', alpha=0.75 ,label="nodes (n)")
  h,l=ax1.get_legend_handles_labels() # get labels and handles from ax1
  ax1.legend(h,l)
  simpleaxis(ax1)
  ax1.set_ylabel('Count per State')

  ax2.plot(df.index,df['cs_m'],color='b',alpha=0.75, label="m")
  ax2.plot(df.index,df['cs_n'],color='r',alpha=0.75, label="n")
  h,l=ax2.get_legend_handles_labels() # get labels and handles from ax1
  ax2.legend(h,l)
  simpleaxis(ax2)
  ax2.set_ylabel('Cumulative Sum')
  # ax2.set_xscale('log')
  ax3.plot(range(0,len(avgk_d)), avgk_d.values(), '.',linestyle="-", alpha=0.5, label=r'$\langle k \rangle$')
  # h,l=ax3.get_legend_handles_labels() # get labels and handles from ax1
  # ax3.legend(h,l)
  ax3.spines['top'].set_visible(False)
  ax3.spines['right'].set_visible(False)
  ax3.set_ylabel(r'Mean degree $\langle k \rangle$')
  ax3.grid(True)

  return

def time_range_each_network_spans(pd_df, graph_name):
  # if graph_name_arg is None:
  #   graph_name_arg = "Contact"
  ''' Find the smallest timestamp delta '''
  print '-'*8,'smallest timestamp delta','-'*8
  if g_name == 'slashdot':
    pd_df = pd_df[pd_df['ts']>0]
  span= pd.to_datetime(df['ts'].max(),unit='s') - pd.to_datetime(df['ts'].min(),unit='s')
  print span

def plot_degree_distribution(pd_df, graph_name):
    ''' Plots Degree Distribution (Count) '''
    if g_name == 'slashdot':
      pd_df = pd_df[pd_df['ts']>0]

    gs = gridspec.GridSpec(1, 1)
    axs =  plt.subplot(gs[0, 0])
    oG = nx.from_pandas_dataframe(df, 'src', 'trg', ['ts'])
    degrees = oG.degree().items()
    ddf = pd.DataFrame.from_dict(degrees)
    gb =  ddf.groupby([1]).count()
    axs.scatter(gb.index.values, gb[0].values)
    gb.plot(ax=axs)
    axs.set_xscale('log')
    axs.set_yscale('log')

def derive_prules_from(list_of_graphs):
    lst_prod_rules = []
    for g in list_of_graphs:
        if g.number_of_nodes() >0:
          pr = PHRG.probabilistic_hrg_deriving_prod_rules(g)
          lst_prod_rules.append(pr)
    return lst_prod_rules

def get_prod_rules(data_frame, nbr_blocks):
    df = data_frame
    nb = int(nbr_blocks)
    chunked_graphs_lst = []
    if nb:
        slice = int ((df.ts.max() - df.ts.min())/nb)
    WG = nx.from_pandas_dataframe(df, 'src', 'trg', ['ts']) # whole graph
    pos = nx.spring_layout(WG)


    for blk in range(df.ts.min(), df.ts.max(), slice):
        mask = (df['ts'] >= blk) & (df['ts'] <= blk+slice)
        ldf = df.loc[mask]
        G = nx.from_pandas_dataframe(ldf, 'src', 'trg', ['ts'])
        chunked_graphs_lst.append(G)
    prules = derive_prules_from(chunked_graphs_lst)
    df = pd.DataFrame(columns=['rid','lhs', 'rhs','p'])

    for k,r in enumerate(prules):
      #print "{}: {}".format(k, [x for x in r if 'S' in x])# [len(x) for x in lhs if 'S' in x])
      # df = pd.concat ([df, pd.DataFrame([x for x in r], columns=['rid','lhs', 'rhs','p'])])
      bdf = pd.DataFrame([x for x in r], columns=['rid','lhs', 'rhs','p'])
      bdf['lcnt'] = bdf['lhs'].apply(lambda x: len(x))
      bdf['rcnt'] = bdf['rhs'].apply(lambda x: len(x))
      df = pd.concat([df,bdf])
      break

    print df.head()
    # print 'size of the rhs'[len(x) for x in df[df['lhs']=='S']['rhs']]
    # tdf = df[['lhs','rhs']].apply(lambda x: [len(r) for r in x])
    # tdf.columns=['lcnt','rcnt']
    # df =pd.concat([df,tdf],axis=1)
    # print df[['lcnt','rcnt']].describe()
    # # df.boxplot(['lcnt','rcnt'])
    # df.boxplot(by=['lhs','rhs'], notch=True)
    # # ax.set_xticks(range(10))
    df.plot.hist()
    plt.savefig('/tmp/outfig', bbox_inches='tight')
    exit()

    ptsg.plot_timestamped_graphs(chunked_graphs_lst,pos=pos, outfigname="tmp1")

    chunked_graphs_lst = []
    for blk in range(df.ts.min(), df.ts.max(), slice):
        mask = (df['ts'] <= blk+slice)
        ldf = df.loc[mask]
        G = nx.from_pandas_dataframe(ldf, 'src', 'trg', ['ts'])
        chunked_graphs_lst.append(G)
    # plot
    ptsg.plot_timestamped_graphs(chunked_graphs_lst, pos=pos, outfigname="tmp2")

    if 0:
        print
        for k,pr in enumerate(prules): ## print enum rules
            print "{}\t{}".format(k,pr)

def get_parser():
    parser = argparse.ArgumentParser(description='Growing: process given edgelist')
    parser.add_argument('-g','--gname', metavar='GRAPH_NAME',required=True, help='graph name to process')
    parser.add_argument('--min_time', action="store_true", default=False)
    parser.add_argument('--accum_nodes',  action="store_true", default=False)
    parser.add_argument('--rules', action="store_true", default=False, help='nbr of rules per T')
    parser.add_argument('--time_range', action="store_true", default=False) # evol_struc_blocks
    parser.add_argument('--evol_struc_blocks', action="store_true", default=False)
    parser.add_argument('--evolving_structure', action="store_true", default=False)
    # parser.add_argument('--min_time', action='min_time', help='Output min time interval across the whole data')
    parser.add_argument('-b','--blocks', required=False,help='Number of blocks or time slices')
    parser.add_argument('--version', action='version', version=__version__)
    return parser

if __name__ == '__main__':
  parser = get_parser()
  args = vars(parser.parse_args())

  in_file = args['gname']
  datframes = tdf.Pandas_DataFrame_From_Edgelist([in_file])
  df = datframes[0]

  # g_name = os.path.basename(in_file).lstrip('out.')
  g_name = os.path.basename(in_file).split('.')[1]
  if len(g_name)>=8:
    g_name=g_name[:8]

  print '...', g_name

  if args['rules']:
    get_prod_rules(df, args['blocks'])
    sys.exit(0)

  if args['min_time']: min_time_info(df, g_name)
  elif args['accum_nodes']: accumulation_of_nodes(df, g_name, out2pdf=False)
  elif args['time_range']: time_range_each_network_spans(df, g_name)
  elif args['evolving_structure']: plot_evolving_structure(df, g_name)
  elif args['evol_struc_blocks']:
    nblocks = args['blocks']
    print int(nblocks)
    plot_evolving_structure_inblocks(df, int(nblocks), g_name)
  elif args['degree']: plot_degree_distribution(df, g_name)
  else:
    sys.exit(0)

  if 0:
    # Save plots
    plt.savefig('/tmp/outfig_'+g_name, bbox_inches='tight')
    print 'saved fig'
    sys.exit(0)


  # ''' Find the smallest timestamp delta '''
  # print '-'*20,'Find the smallest timestamp delta','-'*20
  # if g_name == 'slashdot':
  #   df = df[df['ts']>0]
  # slc_in_secs = find_smallest_timestamp_delta(df,tslices)
  #
  # k = 0
  # unix_epoch = {3600: 'hrs', 86400: 'days', 604800: 'weeks', 2629743: 'months', 31556926: 'years'}
  # print 'Plottting...', slc_in_secs
  # gs  = gridspec.GridSpec(len(slc_in_secs), 1)
  # label_info = []
  # for j,min_tslice in enumerate(slc_in_secs):
  #   axh = plt.subplot(gs[j, 0])
  #   edges_dict = {}
  #
  #   for blk in range(df.ts.min(),df.ts.max(),min_tslice):
  #     mask = (df['ts'] >= blk) & (df['ts'] <= blk+min_tslice)
  #     ldf = df.loc[mask]
  #     G = nx.from_pandas_dataframe(ldf, 'src', 'trg', ['ts'])
  #
  #     clqs = nx.find_cliques(G)
  #     clqs_elem_len = [len(c) for c in clqs]
  #     if len(clqs_elem_len):
  #       edges_dict[blk] = (G.number_of_edges(),len(clqs_elem_len))
  #     else:
  #       edges_dict[blk] = (G.number_of_edges(),0)
  #     # if k>100: break
  #     # k += 1
  #     # print '.',
  #   nf  = pd.DataFrame.from_dict(edges_dict.items())
  #   nf['ecnt'] = nf[1].apply(lambda x: x[0])
  #   nf['clqcnt'] = nf[1].apply(lambda x: x[1])
  #   print nf.head()
  #   nf.plot.scatter(x=[0],y=['ecnt'],   ax=axh, color='r', edgecolors='w', alpha=0.8,label=r"$|edges|_{avg}$")
  #   nf.plot.scatter(x=[0],y=['clqcnt'], ax=axh, color='b', edgecolors='w', alpha=0.8,label=r"$|cliques|_{avg}$")
  #
  #   if min_tslice in unix_epoch.keys():
  #     print unix_epoch[min_tslice]
  #     axh.set_xlabel(unix_epoch[min_tslice])
  #   else:
  #     axh.set_xlabel('{} sec slices'.format(min_tslice))
  # exit()
  # axs0.plot(range(0,len(gb)), gb['w'].values, 'o', alpha=0.2)
  # axs0.set_title(r'Baseline (ignore time, train and predict on the complete graph)')
  # axs0.set_xscale('log')     #open(results_folder+'.done/'+g_gname,'a').close()
  #
  # plt.savefig('/tmp/outfig', bb_inches='tight')
  #
  # # gs = gridspec.GridSpec(tslices,1)
  # # axs0 = plt.subplot(gs[0, :])
  #
  # # Plot evolving avg degree structure of the graph
  # # axs = plt.subplot(gs[2,:])
  # tdeltas = [672,168,24,12]
  #
  # if 0: find_smallest_timestamp_delta(df,tslices)
  #
  # gs = gridspec.GridSpec(len(tdeltas),1)
  # for k,dt in enumerate(tdeltas):
  #   print k
  #   axh = plt.subplot(gs[k,0])
  #   if k==0: axh.set_ylabel('Avg Degree')
  #   Structure_Varying_Overtime(df, dt, axh)
  #   plt.legend()
  #
  #
  #
  #
  #
  # if 0:
  #   # Plot as Unique nodes accumulate ------------------
  #   plot_node_accumulation(df,axs0)
  #
  #   # Plot edges in the dataset ------------------
  #   plot_number_of_edges(df,axs0)
  #
  #   # Plot evolving node count
  #   for t in range(0,tslices):
  #     axh = plt.subplot(gs[1,t])
  #     Growing_Network_Nodes_Edges(df, tslices, t, axh)
  #     break
  #
  #
  #
  # if 0:
  #   # Plot the groups ----------------------------------
  #   for t in range(0,tslices):
  #     axh = plt.subplot(gs[1,t])
  #     plot_kthslice_number_of_edges(df,axh,t, tslices)
  #
  #   # Plot the # of rules per TS -----------------------
  #   for t in range(0,tslices):
  #     axh = plt.subplot(gs[2,t])
  #     plot_nbr_prod_rules_per_ts(df, axh, t, tslices)
  #
  #   ax1 = plt.subplot(gs[1, 0])
  #   ax1.set_ylabel('Edge Count')
  #   ax1 = plt.subplot(gs[2, 0])
  #   ax1.set_ylabel('Avg Clique Length')
  #
  #   # Plot Grow the network using the final State of the Graph
  #   G = nx.from_pandas_dataframe(df, 'src', 'trg', ['ts']) # learn rules on final state
  #   prod_rules = PHRG.probabilistic_hrg_learning(G) # Derive the prod rules
  #
  #   # for t in range(0,tslices):
  #   #   axh = plt.subplot(gs[1,t])
  #   #   Growing_Network_Using_Final_State_ProdRules(df, prod_rules, tslices, t, axh)\
  #   # Growing_Network_Using_Final_State_ProdRules(df, prod_rules, tslices, 3, axs1)
