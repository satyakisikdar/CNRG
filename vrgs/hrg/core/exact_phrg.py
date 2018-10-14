#!/usr/bin/env python
# make the other metrics work
# generate the txt files, then work on the pdf otuput
# TODO:load_edgelist

__version__ = "0.1.0"
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt
import sys
import os
import re
import cPickle
import time
import networkx as nx
import PHRG as phrg
import load_edgelist_from_dataframe as tdf
import tree_decomposition as td
import probabilistic_cfg as pcfg
import net_metrics as metrics
import pprint as pp
import argparse, traceback
import graph_sampler as gs
from multiprocessing import Process
import multiprocessing as mp
import load_edgelist_from_dataframe as tdf

DBG = False
results = []

# ~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~100
def get_parser ():
	parser = argparse.ArgumentParser(description='Infer a model given a graph (derive a model)')
	parser.add_argument('--orig', required=True, nargs=1, help='Filename of edgelist graph')
	parser.add_argument('--chunglu', help='Generate chunglu graphs', action='store_true')
	parser.add_argument('--kron', help='Generate Kronecker product graphs', action='store_true')
	parser.add_argument('--samp', help='Sample sg>dur>gg2targetN', action='store_true')
	parser.add_argument('-tw', action='store_true', default=0, required=0, help="print xphrg mcs tw")
	parser.add_argument('--nstats', action='store_true', default=0, required=0, help="Compute stats?")
	parser.add_argument('-prs', action='store_true', default=0, required=0, help="stop at prs")
	parser.add_argument('--version', action='version', version=__version__)
	return parser

def collect_results(result):
	results.extend(result)


def Hstar_Graphs_Control (G, graph_name, axs=None):
	# Derive the prod rules in a naive way, where
	prod_rules = phrg.probabilistic_hrg_learning(G)
	pp.pprint(prod_rules)
	g = pcfg.Grammar('S')
	for (id, lhs, rhs, prob) in prod_rules:
		g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

	num_nodes = G.number_of_nodes()

	print "Starting max size", 'n=', num_nodes
	g.set_max_size(num_nodes)

	print "Done with max size"

	Hstars = []

	num_samples = 20
	print '*' * 40
	for i in range(0, num_samples):
		rule_list = g.sample(num_nodes)
		hstar = phrg.grow(rule_list, g)[0]
		Hstars.append(hstar)

	# if 0:
	#	 g = nx.from_pandas_dataframe(df, 'src', 'trg', edge_attr=['ts'])
	#	 draw_degree_whole_graph(g,axs)
	#	 draw_degree(Hstars, axs=axs, col='r')
	#	 #axs.set_title('Rules derived by ignoring time')
	#	 axs.set_ylabel('Frequency')
	#	 axs.set_xlabel('degree')

	if 0:
		# metricx = [ 'degree','hops', 'clust', 'assort', 'kcore','eigen','gcd']
		metricx = ['clust']
		# g = nx.from_pandas_dataframe(df, 'src', 'trg',edge_attr=['ts'])
		# graph_name = os.path.basename(f_path).rstrip('.tel')
		if DBG: print ">", graph_name
		metrics.network_properties([G], metricx, Hstars, name=graph_name, out_tsv=True)


def pandas_dataframes_from_edgelists (el_files):
	if (el_files is None):	return
	list_of_dataframes = []
	for f in el_files:
		print '~' * 80
		print f
		temporal_graph = False
		with open(f, 'r') as ifile:
			line = ifile.readline()
			while (not temporal_graph):
				if ("%" in line):
					line = ifile.readline()
				elif len(line.split()) > 3:
					temporal_graph = True
		if (temporal_graph):
			dat = np.genfromtxt(f, dtype=np.int64, comments='%', delimiter="\t", usecols=[0, 1, 3],
													autostrip=True)
			df = pd.DataFrame(dat, columns=['src', 'trg', 'ts'])
		else:
			dat = np.genfromtxt(f, dtype=np.int64, comments='%', delimiter="\t", usecols=[0, 1],
													autostrip=True)
			df = pd.DataFrame(dat, columns=['src', 'trg'])
		df = df.drop_duplicates()
		list_of_dataframes.append(df)

		list_of_dataframes.append(df)

	return list_of_dataframes


def grow_exact_size_hrg_graphs_from_prod_rules (prod_rules, gname, n, runs=1):
	"""
	Args:
		rules: production rules (model)
		gname: graph name
		n:		 target graph order (number of nodes)
		runs:	how many graphs to generate

	Returns: list of synthetic graphs

	"""
	DBG = True
	if n <= 0: sys.exit(1)

	g = pcfg.Grammar('S')
	for (id, lhs, rhs, prob) in prod_rules:
		g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

	print
	print "Added rules HRG (pr", len(prod_rules), ", n,", n, ")"

	num_nodes = n
	if DBG: print "Starting max size ..."
	t_start = time.time()
	g.set_max_size(num_nodes)
	print "Done with max size, took %s seconds" % (time.time() - t_start)

	hstars_lst = []
	print "	",
	for i in range(0, runs):
		print '>',
		rule_list = g.sample(num_nodes)
		hstar = phrg.grow(rule_list, g)[0]
		hstars_lst.append(hstar)

	return hstars_lst


def pwrlaw_plot (xdata, ydata, yerr):
	from scipy import linspace, randn, log10, optimize, sqrt

	powerlaw = lambda x, amp, index: amp * (x ** index)

	logx = log10(xdata)
	logy = log10(ydata)
	logyerr = yerr / ydata

	# define our (line) fitting function
	fitfunc = lambda p, x: p[0] + p[1] * x
	errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

	pinit = [1.0, -1.0]
	out = optimize.leastsq(errfunc, pinit,
												 args=(logx, logy, logyerr), full_output=1)

	pfinal = out[0]
	covar = out[1]
	print pfinal
	print covar

	index = pfinal[1]
	amp = 10.0 ** pfinal[0]

	indexErr = sqrt(covar[0][0])
	ampErr = sqrt(covar[1][1]) * amp

	print index

	# ########
	# plotting
	# ########
	# ax.plot(ydata)
	# ax.plot(pl_sequence)

	fig, axs = plt.subplots(2, 1)

	axs[0].plot(xdata, powerlaw(xdata, amp, index))	# Fit
	axs[0].errorbar(xdata, ydata, yerr=yerr, fmt='k.')	# Data
	(yh1, yh2) = (axs[0].get_ylim()[1] * .9, axs[0].get_ylim()[1] * .8)
	xh = axs[0].get_xlim()[0] * 1.1
	print axs[0].get_ylim()
	print (yh1, yh2)

	axs[0].text(xh, yh1, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr))
	axs[0].text(xh, yh2, 'Index = %5.2f +/- %5.2f' % (index, indexErr))
	axs[0].set_title('Best Fit Power Law')
	axs[0].set_xlabel('X')
	axs[0].set_ylabel('Y')
	# xlim(1, 11)
	#
	# subplot(2, 1, 2)
	axs[1].loglog(xdata, powerlaw(xdata, amp, index))
	axs[1].errorbar(xdata, ydata, yerr=yerr, fmt='k.')	# Data
	axs[1].set_xlabel('X (log scale)')
	axs[1].set_ylabel('Y (log scale)')

	import datetime
	figfname = datetime.datetime.now().strftime("%d%b%y") + "_pl"
	plt.savefig(figfname, bbox_inches='tight')
	return figfname


def deg_vcnt_to_disk (orig_graph, synthetic_graphs):
	df = pd.DataFrame(orig_graph.degree().items())
	gb = df.groupby([1]).count()
	# gb.to_csv("Results/deg_orig_"+orig_graph.name+".tsv", sep='\t', header=True)
	gb.index.rename('k', inplace=True)
	gb.columns = ['vcnt']
	gb.to_csv("Results/deg_orig_" + orig_graph.name + ".tsv", sep='\t', header=True)
	# ## - group of synth graphs -
	deg_df = pd.DataFrame()
	for g in synthetic_graphs:
		d = g.degree()
		df = pd.DataFrame.from_dict(d.items())
		gb = df.groupby(by=[1]).count()
		# Degree vs cnt
		deg_df = pd.concat([deg_df, gb], axis=1)	# Appends to bottom new DFs
	# print gb
	deg_df['mean'] = deg_df.mean(axis=1)
	deg_df.index.rename('k', inplace=True)
	deg_df['mean'].to_csv("Results/deg_xphrg_" + orig_graph.name + ".tsv", sep='\t', header=True)


def plot_g_hstars (orig_graph, synthetic_graphs):
	df = pd.DataFrame(orig_graph.degree().items())
	gb = df.groupby([1]).count()
	# gb.to_csv("Results/deg_orig_"+orig_graph.name+".tsv", sep='\t', header=True)
	gb.index.rename('k', inplace=True)
	gb.columns = ['vcnt']

	# k_cnt = [(x.tolist(),y.values[0]) for x,y in gb.iterrows()]
	xdata = np.array([x.tolist() for x, y in gb.iterrows()])
	ydata = np.array([y.values[0] for x, y in gb.iterrows()])
	yerr = ydata * 0.000001

	fig, ax = plt.subplots()
	ax.plot(gb.index.values, gb['vcnt'].values, '-o', markersize=8, markerfacecolor='w',
					markeredgecolor=[0, 0, 1], alpha=0.5, label="orig")

	ofname = pwrlaw_plot(xdata, ydata, yerr)
	if os.path.exists(ofname): print '... Plot save to:', ofname

	deg_df = pd.DataFrame()
	for g in synthetic_graphs:
		d = g.degree()
		df = pd.DataFrame.from_dict(d.items())
		gb = df.groupby(by=[1]).count()
		# Degree vs cnt
		deg_df = pd.concat([deg_df, gb], axis=1)	# Appends to bottom new DFs
	# print gb
	deg_df['mean'] = deg_df.mean(axis=1)
	deg_df.index.rename('k', inplace=True)
	# ax.plot(y=deg_df.mean(axis=1))
	# ax.plot(y=deg_df.median(axis=1))
	# ax.plot()
	# orig
	deg_df.mean(axis=1).plot(ax=ax, label='mean', color='r')
	deg_df.median(axis=1).plot(ax=ax, label='median', color='g')
	ax.fill_between(deg_df.index, deg_df.mean(axis=1) - deg_df.sem(axis=1),
									deg_df.mean(axis=1) + deg_df.sem(axis=1), alpha=0.2, label="se")
	# ax.plot(k_cnt)
	# deg_df.plot(ax=ax)
	# for x,y in k_cnt:
	#		 if DBG: print "{}\t{}".format(x,y)
	#
	#
	# for g in synths:
	#		 df = pd.DataFrame(g.degree().items())
	#		 gb = df.groupby([1]).count()
	#		 # gb.plot(ax=ax)
	#		 for x,y in k_cnt:
	#				 if DBG: print "{}\t{}".format(x,y)
	#
	# # Curve-fit
	#
	plt.savefig('tmpfig', bbox_inches='tight')


def treewidth (parent, children, twlst):
	twlst.append(parent)
	for x in children:
		if isinstance(x, (tuple, list)):
			treewidth(x[0], x[1], twlst)
		else:
			print type(x), len(x)


def print_treewdith (tree):
	root, children = tree
	print " computing tree width"
	twdth = []
	treewidth(root, children, twdth)
	print '		Treewidth:', np.max([len(x) - 1 for x in twdth])


def get_hrg_production_rules (edgelist_data_frame, graph_name, tw=False,
															n_subg=2, n_nodes=300, nstats=False):
	from growing import derive_prules_from

	t_start = time.time()
	df = edgelist_data_frame
	if df.shape[1] == 4:
		G = nx.from_pandas_dataframe(df, 'src', 'trg', edge_attr=True)	# whole graph
	elif df.shape[1] == 3:
		G = nx.from_pandas_dataframe(df, 'src', 'trg', ['ts'])	# whole graph
	else:
		G = nx.from_pandas_dataframe(df, 'src', 'trg')
	G.name = graph_name
	print "==> read in graph took: {} seconds".format(time.time() - t_start)

	G.remove_edges_from(G.selfloop_edges())
	giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
	G = nx.subgraph(G, giant_nodes)

	num_nodes = G.number_of_nodes()

	phrg.graph_checks(G)

	if DBG: print
	if DBG: print "--------------------"
	if not DBG: print "-Tree Decomposition-"
	if DBG: print "--------------------"

	prod_rules = {}
	K = n_subg
	n = n_nodes
	if num_nodes >= 500:
		print 'Grande'
		t_start = time.time()
		for Gprime in gs.rwr_sample(G, K, n):
			T = td.quickbb(Gprime)
			root = list(T)[0]
			T = td.make_rooted(T, root)
			T = phrg.binarize(T)
			root = list(T)[0]
			root, children = T
			# td.new_visit(T, G, prod_rules, TD)
			td.new_visit(T, G, prod_rules)
			Process(target=td.new_visit, args=(T, G, prod_rules,)).start()
	else:
		T = td.quickbb(G)
		root = list(T)[0]
		T = td.make_rooted(T, root)
		T = phrg.binarize(T)
		root = list(T)[0]
		root, children = T
		# td.new_visit(T, G, prod_rules, TD)
		td.new_visit(T, G, prod_rules)

		print_treewidth(T)
		exit()

	if DBG: print
	if DBG: print "--------------------"
	if DBG: print "- Production Rules -"
	if DBG: print "--------------------"

	for k in prod_rules.iterkeys():
		if DBG: print k
		s = 0
		for d in prod_rules[k]:
			s += prod_rules[k][d]
		for d in prod_rules[k]:
			prod_rules[k][d] = float(prod_rules[k][d]) / float(
				s)	# normailization step to create probs not counts.
			if DBG: print '\t -> ', d, prod_rules[k][d]

	rules = []
	id = 0
	for k, v in prod_rules.iteritems():
		sid = 0
		for x in prod_rules[k]:
			rhs = re.findall("[^()]+", x)
			rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
			if DBG: print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
			sid += 1
		id += 1

	df = pd.DataFrame(rules)
	'''print "++++++++++"
	df.to_csv('ProdRules/{}_prs.tsv'.format(G.name), header=False, index=False, sep="\t")
	if os.path.exists('ProdRules/{}_prs.tsv'.format(G.name)): 
		print 'Saved', 'ProdRules/{}_prs.tsv'.format(G.name)
	else:
		print "Trouble saving"
	print "-----------"
	print [type(x) for x in rules[0]] '''

	'''
	Graph Generation of Synthetic Graphs
	Grow graphs usigng the union of rules from sampled sugbgraphs to predict the target order of the 
	original graph
	'''
	hStars = grow_exact_size_hrg_graphs_from_prod_rules(rules, graph_name, G.number_of_nodes(), 10)
	print '... hStart graphs:', len(hStars)
	d = {graph_name + "_hstars": hStars}
	with open(r"Results/{}_hstars.pickle".format(graph_name), "wb") as output_file:
		cPickle.dump(d, output_file)
	if os.path.exists(r"Results/{}_hstars.pickle".format(graph_name)): print "File saved"

	'''if nstats:
		metricx = ['clust']
		metrics.network_properties([G], metricx, hStars, name=graph_name, out_tsv=False)
	'''

def compute_net_stats_on_read_hrg_pickle(orig_df, gn,metricx):
	with open(r"Results/{}_hstars.pickle".format(gn), "rb") as in_file:
		c = cPickle.load(in_file)
	print " ==> pickle file loaded"
	if isinstance(c, dict):
		if len(c.keys()) == 1:
			c = c.values()[0] # we have k nx gobjects
		else:
			print c.keys()
	if len(orig_df.columns)>=3:
		orig = nx.from_pandas_dataframe(orig_df, 'src', 'trg', edge_attr=['ts'])
	else:
		orig = nx.from_pandas_dataframe(orig_df, 'src', 'trg') 

	# metrics.network_properties([orig], metricx, c, name=gn, out_tsv=False)
	## --
	p = mp.Pool(processes=10)
	for j,gnx in enumerate(c):
		if isinstance (gnx, list):
			gnx = gnx[0]
		p.apply_async(metrics.network_properties, args=([orig], ['clust'], gnx, gn, True, ), callback=collect_results)
	p.close()
	p.join()
	print (results)

def compute_net_statistics_on(orig_df, gn): # gn = graph name
	if gn is "" : return
	print "%"*4, gn, "%"*4
	hrg_pickle_exists = os.path.exists("Results/{}_hstars.pickle".format(gn))
	if hrg_pickle_exists:
		metricx = ['clust']
		compute_net_stats_on_read_hrg_pickle(orig_df, gn,metricx)
	else:
		print 'To gen the hrg pickle:', gn
		exit()


if __name__ == '__main__':
	parser = get_parser()
	args = vars(parser.parse_args())

	# load orig file into DF and get the dataset name into g_name
	datframes = tdf.Pandas_DataFrame_From_Edgelist(args['orig'])
	df = datframes[0]
	g_name = [x for x in os.path.basename(args['orig'][0]).split('.') if len(x) > 3][0]

	if args['chunglu']:
		print 'Generate chunglu graphs given an edgelist'
		sys.exit(0)
	elif args['kron']:
		print 'Generate chunglu graphs given an edgelist'
		sys.exit(0)
	elif (args['nstats']):
		compute_net_statistics_on(df, g_name)
		exit()

	elif args['samp']:
		print 'Sample K subgraphs of n nodes'
		K = 500
		n = 25
		get_hrg_production_rules(df, g_name, n_subg=K, n_nodes=n)
	else:
		try:
			if os.path.exists("Results/{}_hstars.pickle".format(g_name)): 
				print "picle_file already exists"
				os._exit(1)
			else:
				get_hrg_production_rules(df, g_name, args['tw'], nstats=args['nstats'])
		except	Exception, e:
			print 'ERROR, UNEXPECTED SAVE PLOT EXCEPTION'
			print str(e)
			traceback.print_exc()
			os._exit(1)
	sys.exit(0)
