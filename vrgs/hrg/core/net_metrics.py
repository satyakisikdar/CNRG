__author__ = ['Salvador Aguinaga', 'Rodrigo Palacios', 'David Chaing', 'Tim Weninger']

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize':'small',
					'figure.figsize': (1.6 * 10, 1.0 * 10),
					'axes.labelsize': 'small',
					'axes.titlesize': 'small',
					'xtick.labelsize':'small',
					'ytick.labelsize':'small'}
pylab.rcParams.update(params)
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import random
import collections
from collections import Counter
from random import sample
import math

def draw_ugander_graphlet_plot(orig_g, mG, ergm=[], rmat=[]):
		df = pd.DataFrame(mG)
		width = .25
		if len(ergm) > 0:
				dfergm = pd.DataFrame(ergm)
				width = .20

		if len(rmat) > 0:
				rmat = pd.DataFrame(rmat)
				width = .20

		N = 11
		dforig = pd.DataFrame(orig_g)
		means = (dforig.mean()['e0'], dforig.mean()['e1'], dforig.mean()['e2'], dforig.mean()['e2c'], dforig.mean()['tri'],
						 dforig.mean()['p3'], dforig.mean()['star'], dforig.mean()['tritail'], dforig.mean()['square'],
						 dforig.mean()['squarediag'], dforig.mean()['k4'])
		sem = (dforig.sem()['e0'], dforig.sem()['e1'], dforig.sem()['e2'], dforig.sem()['e2c'], dforig.sem()['tri'],
					 dforig.sem()['p3'], dforig.sem()['star'], dforig.sem()['tritail'], dforig.sem()['square'],
					 dforig.sem()['squarediag'], dforig.sem()['k4'])
		ind = np.arange(N)
		fig, ax = plt.subplots()
		print means
		rects = ax.bar(ind + .02, means, width - .02, color='k', yerr=sem)

		means = (df.mean()['e0'], df.mean()['e1'], df.mean()['e2'], df.mean()['e2c'], df.mean()['tri'], df.mean()['p3'],
						 df.mean()['star'], df.mean()['tritail'], df.mean()['square'], df.mean()['squarediag'], df.mean()['k4'])
		sem = (
				df.sem()['e0'], df.sem()['e1'], df.sem()['e2'], df.sem()['e2c'], df.sem()['tri'], df.sem()['p3'],
				df.sem()['star'],
				df.sem()['tritail'], df.sem()['square'], df.sem()['squarediag'], df.sem()['k4'])
		rects = ax.bar(ind + width + .02, means, width - .02, color='b', yerr=sem)
		print means
		ax.set_yscale("log", nonposy='clip')

		if len(ergm) > 0:
				means = (
						dfergm.mean()['e0'], dfergm.mean()['e1'], dfergm.mean()['e2'], dfergm.mean()['e2c'], dfergm.mean()['tri'],
						dfergm.mean()['p3'], dfergm.mean()['star'], dfergm.mean()['tritail'], dfergm.mean()['square'],
						dfergm.mean()['squarediag'], dfergm.mean()['k4'])
				sem = (dfergm.sem()['e0'], dfergm.sem()['e1'], dfergm.sem()['e2'], dfergm.sem()['e2c'], dfergm.sem()['tri'],
							 dfergm.sem()['p3'], dfergm.sem()['star'], dfergm.sem()['tritail'], dfergm.sem()['square'],
							 dfergm.sem()['squarediag'], dfergm.sem()['k4'])
				rects = ax.bar(ind + width + width + width + .02, means, width - .02, color='r', yerr=sem)
				print means

		if len(rmat) > 0:
				means = (rmat.mean()['e0'], rmat.mean()['e1'], rmat.mean()['e2'], rmat.mean()['e2c'], rmat.mean()['tri'],
								 rmat.mean()['p3'], rmat.mean()['star'], rmat.mean()['tritail'], rmat.mean()['square'],
								 rmat.mean()['squarediag'], rmat.mean()['k4'])
				print means
				rects = ax.bar(ind + width + width + .02, means, width - .02, color='purple')

		plt.ylim(ymin=0)
		# fig = plt.gcf()
		# fig.set_size_inches(5, 3, forward=True)
		plt.show()


def hops(all_succs, start, level=0, debug=False):
		if debug: print("level:", level)

		succs = all_succs[start] if start in all_succs else []
		if debug: print("succs:", succs)

		lensuccs = len(succs)
		if debug: print("lensuccs:", lensuccs)
		if debug: print()
		if not succs:
				yield level, 0
		else:
				yield level, lensuccs

		for succ in succs:
				# print("succ:", succ)
				for h in hops(all_succs, succ, level + 1):
						yield h


def get_graph_hops(graph, num_samples):
		c = Counter()
		for i in range(0, num_samples):
				node = sample(graph.nodes(), 1)[0]
				b = nx.bfs_successors(graph, node)

				for l, h in hops(b, node):
						c[l] += h

		hopper = Counter()
		for l in c:
				hopper[l] = float(c[l]) / float(num_samples)
		return hopper


def bfs_eff_diam(G, NTestNodes, P):
		if G.number_of_nodes() == 0:
				return 0

		EffDiam = -1
		FullDiam = -1
		AvgSPL = -1

		DistToCntH = {}

		NodeIdV = nx.nodes(G)
		random.shuffle(NodeIdV)

		for tries in range(0, min(NTestNodes, nx.number_of_nodes(G))):
				NId = NodeIdV[tries]
				b = nx.bfs_successors(G, NId)
				for l, h in hops(b, NId):
						if h is 0: continue
						if not l + 1 in DistToCntH:
								DistToCntH[l + 1] = h
						else:
								DistToCntH[l + 1] += h

		DistNbrsPdfV = {}
		SumPathL = 0.0
		PathCnt = 0.0
		for i in DistToCntH.keys():
				DistNbrsPdfV[i] = DistToCntH[i]
				SumPathL += i * DistToCntH[i]
				PathCnt += DistToCntH[i]

		oDistNbrsPdfV = collections.OrderedDict(sorted(DistNbrsPdfV.items()))

		CdfV = oDistNbrsPdfV
		for i in range(1, len(CdfV)):
				if not i + 1 in CdfV:
						CdfV[i + 1] = 0
				CdfV[i + 1] = CdfV[i] + CdfV[i + 1]

		EffPairs = P * CdfV[next(reversed(CdfV))]

		for ValN in CdfV.keys():
				if CdfV[ValN] > EffPairs: break

		if ValN >= len(CdfV): return next(reversed(CdfV))
		if ValN is 0: return 1
		# interpolate
		DeltaNbrs = CdfV[ValN] - CdfV[ValN - 1];
		if DeltaNbrs is 0: return ValN;
		return ValN - 1 + (EffPairs - CdfV[ValN - 1]) / DeltaNbrs


def draw_diam_plot(orig_g, mG):
		df = pd.DataFrame(mG)
		gD = bfs_eff_diam(orig_g, 20, .9)
		ori_degree_seq = []
		for i in range(0, len(max(mG))):
				ori_degree_seq.append(gD)

		plt.fill_between(df.columns, df.mean() - df.sem(), df.mean() + df.sem(), color='blue', alpha=0.2, label="se")
		h, = plt.plot(df.mean(), color='blue', aa=True, linewidth=4, ls='--', label="H*")
		orig, = plt.plot(ori_degree_seq, color='black', linewidth=2, ls='-', label="H")

		plt.title('Diameter Plot')
		plt.ylabel('Diameter')
		plt.xlabel('Growth')

		plt.tick_params(
				axis='x',	# changes apply to the x-axis
				which='both',	# both major and minor ticks are affected
				bottom='off',	# ticks along the bottom edge are off
				top='off',	# ticks along the top edge are off
				labelbottom='off')	# labels along the bottom edge are off
		plt.legend([orig, h], ['$H$', 'HRG $H^*$'], loc=4)
		# fig = plt.gcf()
		# fig.set_size_inches(5, 4, forward=True)
		plt.show()


def draw_graphlet_plot(orig_g, mG):
		df = pd.DataFrame(mG)
		width = .25

		N = 11
		dforig = pd.DataFrame(orig_g)
		means = (dforig.mean()['e0'], dforig.mean()['e1'], dforig.mean()['e2'], dforig.mean()['e2c'], dforig.mean()['tri'],
						 dforig.mean()['p3'], dforig.mean()['star'], dforig.mean()['tritail'], dforig.mean()['square'],
						 dforig.mean()['squarediag'], dforig.mean()['k4'])
		sem = (dforig.sem()['e0'], dforig.sem()['e1'], dforig.sem()['e2'], dforig.sem()['e2c'], dforig.sem()['tri'],
					 dforig.sem()['p3'], dforig.sem()['star'], dforig.sem()['tritail'], dforig.sem()['square'],
					 dforig.sem()['squarediag'], dforig.sem()['k4'])
		ind = np.arange(N)
		fig, ax = plt.subplots()
		rects = ax.bar(ind + .02, means, width - .02, color='k', yerr=sem)

		means = (df.mean()['e0'], df.mean()['e1'], df.mean()['e2'], df.mean()['e2c'], df.mean()['tri'], df.mean()['p3'],
						 df.mean()['star'], df.mean()['tritail'], df.mean()['square'], df.mean()['squarediag'], df.mean()['k4'])
		sem = (
				df.sem()['e0'], df.sem()['e1'], df.sem()['e2'], df.sem()['e2c'], df.sem()['tri'], df.sem()['p3'],
				df.sem()['star'],
				df.sem()['tritail'], df.sem()['square'], df.sem()['squarediag'], df.sem()['k4'])
		rects = ax.bar(ind + width + .02, means, width - .02, color='b', yerr=sem)

		plt.ylim(ymin=0)
		# fig = plt.gcf()
		# fig.set_size_inches(5, 3, forward=True)
		plt.show()


def draw_degree_rank_plot(orig_g, mG):
		ori_degree_seq = sorted(nx.degree(orig_g).values(), reverse=True)	# degree sequence
		deg_seqs = []
		for newg in mG:
				deg_seqs.append(sorted(nx.degree(newg).values(), reverse=True))	# degree sequence
		df = pd.DataFrame(deg_seqs)

		plt.xscale('log')
		plt.yscale('log')
		plt.fill_between(df.columns, df.mean() - df.sem(), df.mean() + df.sem(), color='blue', alpha=0.2, label="se")
		h, = plt.plot(df.mean(), color='blue', aa=True, linewidth=4, ls='--', label="H*")
		orig, = plt.plot(ori_degree_seq, color='black', linewidth=4, ls='-', label="H")

		plt.title('Degree Distribution')
		plt.ylabel('Degree')
		plt.ylabel('Ordered Vertices')

		plt.tick_params(
				axis='x',	# changes apply to the x-axis
				which='both',	# both major and minor ticks are affected
				bottom='off',	# ticks along the bottom edge are off
				top='off',	# ticks along the top edge are off
				labelbottom='off')	# labels along the bottom edge are off

		plt.legend([orig, h], ['$H$', 'HRG $H^*$'], loc=3)
		# fig = plt.gcf()
		# fig.set_size_inches(5, 4, forward=True)
		plt.show()


def draw_network_value(orig_g_M, chunglu_M, HRG_M, pHRG_M, kron_M):
		"""
		Network values: The distribution of eigenvector components (indicators of "network value")
		associated to the largest eigenvalue of the graph adjacency matrix has also been found to be
		skewed (Chakrabarti et al., 2004).
		"""

		eig_cents = [nx.eigenvector_centrality_numpy(g) for g in orig_g_M]	# nodes with eigencentrality
		net_vals = []
		for cntr in eig_cents:
				net_vals.append(sorted(cntr.values(), reverse=True))
		df = pd.DataFrame(net_vals)

		print "orig"
		l = list(df.mean())
		zz = float(len(l))
		if not zz == 0:
				sa =	int(math.ceil(zz/75))
				for i in range(0, len(l), sa):
						print "(" + str(i) + "," + str(l[i]) + ")"

		eig_cents = [nx.eigenvector_centrality_numpy(g) for g in pHRG_M]	# nodes with eigencentrality
		net_vals = []
		for cntr in eig_cents:
				net_vals.append(sorted(cntr.values(), reverse=True))
		df = pd.DataFrame(net_vals)

		print "phrg"
		l = list(df.mean())
		zz = float(len(l))
		if not zz == 0:
				sa =	int(math.ceil(zz/75))
				for i in range(0, len(l), sa):
						print "(" + str(i) + "," + str(l[i]) + ")"

		eig_cents = [nx.eigenvector_centrality_numpy(g) for g in HRG_M]	# nodes with eigencentrality
		net_vals = []
		for cntr in eig_cents:
				net_vals.append(sorted(cntr.values(), reverse=True))
		df = pd.DataFrame(net_vals)

		print "hrg"
		l = list(df.mean())
		zz = float(len(l))
		if not zz == 0:
				sa =	int(math.ceil(zz/75))
				for i in range(0, len(l), sa):
						print "(" + str(i) + "," + str(l[i]) + ")"

		eig_cents = [nx.eigenvector_centrality_numpy(g) for g in chunglu_M]	# nodes with eigencentrality
		net_vals = []
		for cntr in eig_cents:
				net_vals.append(sorted(cntr.values(), reverse=True))
		df = pd.DataFrame(net_vals)

		print "cl"
		l = list(df.mean())
		zz = float(len(l))
		if not zz == 0:
				sa =	int(math.ceil(zz/75))
				for i in range(0, len(l), sa):
						print "(" + str(i) + "," + str(l[i]) + ")"

		eig_cents = [nx.eigenvector_centrality_numpy(g) for g in kron_M]	# nodes with eigencentrality
		net_vals = []
		for cntr in eig_cents:
				net_vals.append(sorted(cntr.values(), reverse=True))
		df = pd.DataFrame(net_vals)

		print "kron"
		l = list(df.mean())
		zz = float(len(l))
		if not zz == 0:
				sa =	int(math.ceil(zz/75))
				for i in range(0, len(l), sa):
						print "(" + str(i) + "," + str(l[i]) + ")"

def degree_distribution_multiples(graphs):
		if graphs is not None:
			dorig = pd.DataFrame()
			for g in graphs:
					d	= g.degree()
					df = pd.DataFrame.from_dict(d.items())
					gb = df.groupby(by=[1]).count()
					dorig = pd.concat([dorig, gb], axis=1)	# Appends to bottom new DFs

		return dorig

def hop_plot_multiples(graphs):
		if graphs is not None:
				m_hops_ar = []
				for g in graphs:
						c = get_graph_hops(g, 20)
						d = dict(c)
						m_hops_ar.append(d.values())

				hops_df = pd.DataFrame(m_hops_ar)

		return hops_df.transpose()

def clustering_coefficients_single(graph):
	g = graph
	dorig = pd.DataFrame()
	degdf = pd.DataFrame.from_dict(g.degree().items())
	ccldf = pd.DataFrame.from_dict(nx.clustering(g).items())
	dat = np.array([degdf[0], degdf[1], ccldf[1]])
	df 	= pd.DataFrame(np.transpose(dat))
	df 	= df.astype(float)
	df.columns = ['v', 'k', 'cc']
	#dorig = pd.concat([dorig, df])
	return df

def clustering_coefficients_multiples(graphs):
		if graphs is not None:
			dorig = pd.DataFrame()
			for g in graphs:
					degdf = pd.DataFrame.from_dict(g.degree().items())
					ccldf = pd.DataFrame.from_dict(nx.clustering(g).items())

					dat = np.array([degdf[0], degdf[1], ccldf[1]])
					df = pd.DataFrame(np.transpose(dat))
					df = df.astype(float)
					df.columns = ['v', 'k', 'cc']

					dorig = pd.concat([dorig, df])	# Appends to bottom new DFs
		return dorig

def assortativity_coefficients_multiples(graphs):
		if len(graphs) is not 0:
				dorig = pd.DataFrame()
				for g in graphs:
						kcdf = pd.DataFrame.from_dict(nx.average_neighbor_degree(g).items())
						kcdf['k'] = g.degree().values()
						dorig = pd.concat([dorig, kcdf])
		return dorig

def kcore_decomposition_multiples(graphs):
		dorig = pd.DataFrame()
		for g in graphs:
				g.remove_edges_from(g.selfloop_edges())
				d = nx.core_number(g)
				df = pd.DataFrame.from_dict(d.items())
				df[[0]] = df[[0]].astype(int)
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs

		return dorig

def eigenvector_multiples(graphs):
		#
		#	dorig = pd.DataFrame()
		# for g in graphs:
		#		 # d = nx.eigenvector_centrality(g)
		#		 d = nx.eigenvector_centrality_numpy(g)
		#		 df = pd.DataFrame.from_dict(d.items())
		#		 gb = df.groupby(by=[1])
		#		 dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		# # print "orig"
		# # print dorig.mean(axis=1)
		eig_cents = [nx.eigenvector_centrality_numpy(g) for g in graphs]	# nodes with eigencentrality
		net_vals = []
		for cntr in eig_cents:
				net_vals.append(sorted(cntr.values(), reverse=True))
		df = pd.DataFrame(net_vals)

		return df

def network_properties(orig, net_mets, synth_graphs_lst, name='', out_tsv=False):
		'''
		compute network properties
		orig:			 original graph
		net_mets:	 network metrics list
		graphs_lst: graphs to compute degree
		out_tsv:		if True output tsv file for PGFPlots
		'''
		results_d = {}
		gs = gridspec.GridSpec(3,3)
		ax0 = plt.subplot(gs[0, :])
		ax1 = plt.subplot(gs[1, 0])
		ax2 = plt.subplot(gs[1, 1])
		ax3 = plt.subplot(gs[1, 2])
		ax4 = plt.subplot(gs[2, 0])
		ax5 = plt.subplot(gs[2, 1])
		ax6 = plt.subplot(gs[2, 2])

		plt.suptitle(name)

		import os
		if not os.path.exists('./Results'):
			os.makedirs('./Results')

		if 'degree' in net_mets:
				print 'Degree'
				orig__Deg = degree_distribution_multiples(orig)
				orig__Deg.mean(axis=1).plot(ax=ax0,marker='.', ls="None", markeredgecolor="w", color='b')
				synth_Deg = degree_distribution_multiples(synth_graphs_lst)
				synth_Deg.to_csv('Results/degree_synth_{}.tsv'.format(name),sep='\t',header=None, index=False)
				if os.path.exists('Results/degree_synth_{}.tsv'.format(name)): print 'saved to disk'

				synth_Deg.mean(axis=1).plot(ax=ax0,alpha=0.5, color='r')
				synth_Deg.max(axis=1).plot(ax=ax0,alpha=0.2, color='r')
				synth_Deg.min(axis=1).plot(ax=ax0,alpha=0.2, color='r')
				orig__Deg.mean(axis=1).to_csv('Results/degree_orig_{}.tsv'.format(name),sep='\t')
				synth_Deg.mean(axis=1).to_csv('Results/degree_hrg_{}.tsv'.format(name),sep='\t')
				ax0.set_title('Degree distributuion', y=0.9)
				#ax0.set_xscale('log')
				#ax0.set_yscale('log')
				xdat = synth_Deg.index.values
				ydat = synth_Deg.median(axis=1).values
				zdat = synth_Deg.std(axis=1).values
				df1 = pd.DataFrame()
				df1['xdat'] = xdat
				df1['ydat'] = ydat
				df1['ysig'] = zdat
				# df2 = pd.DataFrame()
				# df2['s_med'] = zdat
				# df2['s_std'] = wdat
				# df = df1.join(df2, how='outer')
				df1.to_csv('Results/deg_dist_{}.tsv'.format(name),sep='\t', header=None, index=False)
				if os.path.exists('Results/deg_dist_{}.tsv'.format(name)):
						print '... file written:','Results/deg_dist_{}.tsv'.format(name)

		if 'hops' in net_mets:
			print 'Hops'
			orig__Hop_Plot = hop_plot_multiples(orig)
			synth_Hop_Plot = hop_plot_multiples(synth_graphs_lst)
			orig__Hop_Plot.mean(axis=1).plot(ax=ax1, marker='o', color='b')
			synth_Hop_Plot.mean(axis=1).plot(ax=ax1, color='r')
			synth_Hop_Plot.max(axis=1).plot(ax=ax1, color='r', alpha=0.2)
			synth_Hop_Plot.min(axis=1).plot(ax=ax1, color='r', alpha=0.2)
			ax1.set_title('Hop Plot', y=0.9)

			if out_tsv: orig__Hop_Plot.mean(axis=1).to_csv('Results/hops_orig_{}.tsv'.format(name),sep='\t')
			if out_tsv: synth_Hop_Plot.mean(axis=1).to_csv('Results/hops_hrg_{}.tsv'.format(name),sep='\t')
			print (orig__Hop_Plot.mean(axis=1).to_string())
			print (synth_Hop_Plot.mean(axis=1).to_string())

		if 'clust' in net_mets:
			print 'Clustering Coef'
			orig__clust_coef = clustering_coefficients_multiples(orig)
			synth_clust_coef = clustering_coefficients_multiples(synth_graphs_lst)

			gb = orig__clust_coef.groupby(['k'])
			gb['cc'].mean().plot(ax=ax2, marker='o', ls="None", markeredgecolor="w", color='b', alpha=0.8)
			if out_tsv:
				gb['cc'].mean().to_csv('Results/clust_orig_{}.tsv'.format(name),sep='\t')
			print (gb['cc'].mean().to_string())
			results_d['clust'] = {"orig": list(gb['cc'].mean())}

			gb = synth_clust_coef.groupby(['k'])
			gb['cc'].mean().plot(ax=ax2, marker='o', ls="None", markeredgecolor="w", color='r',	alpha=0.8 )
			ax2.set_title('Avg Clustering Coefficient', y=0.9)
			if out_tsv:
				gb['cc'].mean().to_csv('Results/clust_hrg_{}.tsv'.format(name),mode='a', sep='\t')
			print (gb['cc'].mean().to_string())
			results_d['clust'] = {"synth": list(gb['cc'].mean())}
			return "end" 

		if 'assort' in net_mets:
			print 'Assortativity'
			orig__assort = assortativity_coefficients_multiples(orig)
			synth_assort = assortativity_coefficients_multiples(synth_graphs_lst)

			gb = orig__assort.groupby(['k'])
			gb[1].mean().plot(ax=ax3, marker='o', ls="None", markeredgecolor="w", color='b',	alpha=0.8 )
			gb[1].mean().to_csv('Results/assort_orig_{}.tsv'.format(name),sep='\t')

			gb = synth_assort.groupby(['k'])
			gb[1].mean().plot(ax=ax3, marker='o', ls="None", markeredgecolor="w", color='r',	alpha=0.8 )
			ax3.set_title('Assortativity', y=0.9)
			gb[1].mean().to_csv('Results/assort_hrg_{}.tsv'.format(name),sep='\t')

		if 'kcore' in net_mets:
			print 'kcore_decomposition'
			orig__kcore = kcore_decomposition_multiples(orig)
			synth_kcore = kcore_decomposition_multiples(synth_graphs_lst)


			orig__kcore.plot(ax=ax4, marker='o', ls="None", markeredgecolor="w", color='b',	alpha=0.8 )
			synth_kcore.mean(axis=1).plot(ax=ax4, marker='o', ls="None", markeredgecolor="w", color='r',	alpha=0.8 )
			synth_kcore.max(axis=1).plot(ax=ax4, color='r',	alpha=0.2 )
			synth_kcore.min(axis=1).plot(ax=ax4, color='r',	alpha=0.2 )
			ax4.set_title('K-Core', y=0.9)

			orig__kcore.to_csv('Results/kcore_orig_{}.tsv'.format(name),sep='\t')
			synth_kcore.mean(axis=1).to_csv('Results/kcore_hrg_{}.tsv'.format(name),sep='\t')

		if 'eigen' in net_mets:
			print 'eigenvec'
			orig__eigenvec = eigenvector_multiples(orig)
			synth_eigenvec = eigenvector_multiples(synth_graphs_lst)

			orig__eigenvec= orig__eigenvec.transpose()
			orig__eigenvec.plot(ax=ax5, marker='o', ls="None", markeredgecolor="w", color='b',	alpha=0.8)
			orig__eigenvec.mean(axis=1).to_csv('Results/eigenv_orig_{}.tsv'.format(name),sep='\t')

			synth_eigenvec= synth_eigenvec.transpose()
			synth_eigenvec.mean(axis=1).plot(ax=ax5, marker='s', ls="None", markeredgecolor="w", color='r',	alpha=0.8)
			synth_eigenvec.mean(axis=1).to_csv('Results/eigenv_hrg_{}.tsv'.format(name),sep='\t')
			ax5.set_title('eigenvector', y=0.9)

		import pprint as pp
		if 'gcd' in net_mets:
			print 'GCD'
			ax6.set_title('GCD', y=0.9)
			gcd_hrg = []
			df_g = external_rage(orig[0],name) # original graph
			for synthG in synth_graphs_lst:
				gcd_network = external_rage(synthG,name)
				# rgfd =	tijana_eval_rgfd(df_g, gcd_network)	## what is this?
				gcm_g = tijana_eval_compute_gcm(df_g)
				gcm_h = tijana_eval_compute_gcm(gcd_network)
				gcd_hrg.append(tijana_eval_compute_gcd(gcm_g, gcm_h))

			gcd_hrg_mean = np.mean(gcd_hrg)
			gcd_hrg_std	= np.std(gcd_hrg)

			ax6.bar([1], gcd_hrg_mean, width=0.5, yerr=gcd_hrg_std)
			# http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
			# ax6.set_xticklabels(['HRG'])	## Custom x-axis labels
			ax6.get_xaxis().tick_bottom() ## Remove top axes and right axes ticks
			ax6.get_yaxis().tick_left()
			ax6.set_xlim(0, 5)
			if out_tsv:
				with open ('Results/gcd_{}.tsv'.format(name), 'w') as f:
					f.write('{}\t{}\n'.format(gcd_hrg_mean,gcd_hrg_std))
			print ('{}\t{}\n'.format(gcd_hrg_mean,gcd_hrg_std))
		"""oufigname = '/tmp/outfig_{}.pdf'.format(name)
		plt.savefig(oufigname, bbox_inches='tight')
		if os.path.exists(oufigname): print 'Output: ',oufigname
		"""

		return results_d 



def draw_degree_probability_distribution(orig_g_M, chunglu_M, HRG_M, pHRG_M, kron_M):
		print 'draw_degree_probability_distribution'

		if orig_g_M is not None:
			dorig = pd.DataFrame()
			for g in orig_g_M:
				d = g.degree()
				df = pd.DataFrame.from_dict(d.items())
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
			print "orig"
			if not dorig.empty :
				zz = len(dorig.mean(axis=1).values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(dorig.mean(axis=1).values), sa):
						print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"

		if HRG_M is not None:
			dorig = pd.DataFrame()
			for g in HRG_M:
				d = g.degree()
				df = pd.DataFrame.from_dict(d.items())
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
			print "hrgm"
			if not dorig.empty :
				zz = len(dorig.mean(axis=1).values)
				sa =	int(math.ceil(zz/75))
				for x in range(0, len(dorig.mean(axis=1).values), sa):
						print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"

		if pHRG_M is not None:
			dorig = pd.DataFrame()
			for g in pHRG_M:
				d = g.degree()
				df = pd.DataFrame.from_dict(d.items())
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
			print "phrgm"
			if not dorig.empty :
				zz = len(dorig.mean(axis=1).values)
				sa =	int(math.ceil(zz/75))
				for x in range(0, len(dorig.mean(axis=1).values), sa):
						print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"

		dorig = pd.DataFrame()
		for g in chunglu_M:
				d = g.degree()
				df = pd.DataFrame.from_dict(d.items())
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		print "cl"
		if not dorig.empty :
				zz = len(dorig.mean(axis=1).values)
				sa = int(math.ceil(zz/float(75)))
				print zz, sa
				for x in range(0, len(dorig.mean(axis=1).values), sa):
						print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"

		dorig = pd.DataFrame()
		#print len(kron_M), kron_M
		for g in kron_M:
				print "---=>",len(g)
				d = g.degree()
				df = pd.DataFrame.from_dict(d.items())
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		print "kron"
		if not dorig.empty :
				zz = len(dorig.mean(axis=1).values)
				sa = int(math.ceil(zz/float(75)))
				for x in range(1, len(dorig.mean(axis=1).values), sa):
						print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"


def draw_eigenvector_probability_distribution(orig_g_M, chunglu_M, HRG_M, pHRG_M, kron_M):
		dorig = pd.DataFrame()
		for g in orig_g_M:
				d = nx.eigenvector_centrality(g)
				df = pd.DataFrame.from_dict(d.items())
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		print "orig"
		print dorig.mean(axis=1)

		dorig = pd.DataFrame()
		for g in HRG_M:
				d = nx.eigenvector_centrality(g)
				df = pd.DataFrame.from_dict(d.items())
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		print "hrgm"
		print dorig.mean(axis=1)

		dorig = pd.DataFrame()
		for g in pHRG_M:
				d = nx.eigenvector_centrality(g)
				df = pd.DataFrame.from_dict(d.items())
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		print "phrgm"
		print dorig.mean(axis=1)

		dorig = pd.DataFrame()
		for g in chunglu_M:
				d = nx.eigenvector_centrality(g)
				df = pd.DataFrame.from_dict(d.items())
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		print "cl"
		print dorig.mean(axis=1)

		dorig = pd.DataFrame()
		for g in kron_M:
				d = nx.eigenvector_centrality(g)
				df = pd.DataFrame.from_dict(d.items())
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		print "kron"
		print dorig.mean(axis=1)


def draw_hop_plot(orig_g_M, chunglu_M, HRG_M, pHRG_M, kron_M):
		m_hops_ar = []
		for g in chunglu_M:
				c = get_graph_hops(g, 20)
				d = dict(c)
				m_hops_ar.append(d.values())
				print "Chung Lu hops finished"
		chunglu_df = pd.DataFrame(m_hops_ar)

		m_hops_ar = []
		for g in HRG_M:
				c = get_graph_hops(g, 20)
				d = dict(c)
				m_hops_ar.append(d.values())
				print "HRG hops finished"
		hrg_df = pd.DataFrame(m_hops_ar)

		m_hops_ar = []
		for g in pHRG_M:
				c = get_graph_hops(g, 20)
				d = dict(c)
				m_hops_ar.append(d.values())
				print "PHRG hops finished"
		phrg_df = pd.DataFrame(m_hops_ar)

		m_hops_ar = []
		for g in kron_M:
				c = get_graph_hops(g, 20)
				d = dict(c)
				m_hops_ar.append(d.values())
				print "Kron hops finished"
		kron_df = pd.DataFrame(m_hops_ar)

		## original plot
		m_hops_ar = []
		for g in orig_g_M:
				c = get_graph_hops(g, 20)
				d = dict(c)
				m_hops_ar.append(d.values())
		dorig = pd.DataFrame(m_hops_ar)

		if 0:
			# plt.fill_between(dorig.columns, dorig.mean() - dorig.sem(), dorig.mean() + dorig.sem(), color='black', alpha=0.2, label="se")
			orig, = plt.plot(dorig.mean(), color='black', marker="o", markersize=10, aa=False, linewidth=3, ls='-', label="H")
			print "Hop plot, BA (256, 3)"
			print "H"
			for x in range(0, len(dorig.mean().values)):
					print "(" + str(dorig.mean().index[x]) + ", " + str(dorig.mean().values[x]) + ")"

			# plt.fill_between(phrg_df.columns, phrg_df.mean() - phrg_df.sem(), phrg_df.mean() + phrg_df.sem(), color='blue', alpha=0.2, label="se")
			phrg_h, = plt.plot(phrg_df.mean(), color='blue', marker="d", aa=False, linewidth=3, ls='-', label="PHRG")
			print "PHRG"
			for x in range(0, len(phrg_df.mean().values)):
					print "(" + str(phrg_df.mean().index[x]) + ", " + str(phrg_df.mean().values[x]) + ")"

			# plt.fill_between(hrg_df.columns, hrg_df.mean() - hrg_df.sem(), hrg_df.mean() + hrg_df.sem(), color='red', alpha=0.2, label="se")
			hrg_h, = plt.plot(hrg_df.mean(), color='red', marker="^", aa=False, linewidth=3, ls='-', label="HRG")
			print "HRG"
			for x in range(0, len(hrg_df.mean().values)):
					print "(" + str(hrg_df.mean().index[x]) + ", " + str(hrg_df.mean().values[x]) + ")"


			# plt.fill_between(chunglu_df.columns, chunglu_df.mean() - chunglu_df.sem(), chunglu_df.mean() + chunglu_df.sem(), color='green', alpha=0.2, label="se")
			cl_h, = plt.plot(chunglu_df.mean(), color='green', marker="v", aa=False, linewidth=3, ls='-', label="Chung-Lu")
		print "CL"
		for x in range(0, len(chunglu_df.mean().values)):
				print "(" + str(chunglu_df.mean().index[x]) + ", " + str(chunglu_df.mean().values[x]) + ")"

		if 0:
			# plt.fill_between(kron_df.columns, kron_df.mean() - kron_df.sem(), kron_df.mean() + kron_df.sem(), color='purple', alpha=0.2, label="se")
			kron_h, = plt.plot(kron_df.mean(), color='purple', marker="s", aa=False, linewidth=3, ls='-', label="Kronecker")
		print "K"
		for x in range(0, len(kron_df.mean().values)):
				print "(" + str(kron_df.mean().index[x]) + ", " + str(kron_df.mean().values[x]) + ")"

		if 0:
			plt.title('Hop Plot')
			plt.ylabel('Reachable Pairs')
			plt.xlabel('Number of Hops')
			# plt.ylim(ymax=max(dorig.values()) + max(dorig.values()) * .10)

			plt.legend([orig, phrg_h, hrg_h, cl_h, kron_h], ['$H$', 'PHRG', 'HRG', 'Chung-Lu', 'Kron'], loc=1)
			# fig = plt.gcf()
			# fig.set_size_inches(5, 4, forward=True)
			plt.show()


def draw_assortativity_coefficients(orig_g_M, chunglu_M, HRG_M, pHRG_M, kron_M):

		if len(orig_g_M) is not 0:
				dorig = pd.DataFrame()
				for g in orig_g_M:
						kcdf = pd.DataFrame.from_dict(nx.average_neighbor_degree(g).items())
						kcdf['k'] = g.degree().values()
						dorig = pd.concat([dorig, kcdf])

				print "orig"
				gb = dorig.groupby(['k'])
				zz = len(gb[1].mean().values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(gb[1].mean().values), sa):
						print "(" + str(gb.mean().index[x]) + ", " + str(gb[1].mean().values[x]) + ")"

		if len(chunglu_M) is not 0:
				dorig = pd.DataFrame()
				for g in chunglu_M:
						kcdf = pd.DataFrame.from_dict(nx.average_neighbor_degree(g).items())
						kcdf['k'] = g.degree().values()
						dorig = pd.concat([dorig, kcdf])

				print "cl"
				gb = dorig.groupby(['k'])
				zz = len(gb[1].mean().values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(gb[1].mean().values), sa):
						print "(" + str(gb.mean().index[x]) + ", " + str(gb[1].mean().values[x]) + ")"

		if len(HRG_M) is not 0:
				dorig = pd.DataFrame()
				for g in HRG_M:
						kcdf = pd.DataFrame.from_dict(nx.average_neighbor_degree(g).items())
						kcdf['k'] = g.degree().values()
						dorig = pd.concat([dorig, kcdf])

				print "hrg"
				gb = dorig.groupby(['k'])
				zz = len(gb[1].mean().values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(gb[1].mean().values), sa):
						print "(" + str(gb.mean().index[x]) + ", " + str(gb[1].mean().values[x]) + ")"

		if len(pHRG_M) is not 0:
				dorig = pd.DataFrame()
				for g in pHRG_M:
						kcdf = pd.DataFrame.from_dict(nx.average_neighbor_degree(g).items())
						kcdf['k'] = g.degree().values()
						dorig = pd.concat([dorig, kcdf])

				print "phrg"
				gb = dorig.groupby(['k'])
				zz = len(gb[1].mean().values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(gb[1].mean().values), sa):
						print "(" + str(gb.mean().index[x]) + ", " + str(gb[1].mean().values[x]) + ")"

		if len(kron_M) is not 0:
				dorig = pd.DataFrame()
				for g in kron_M:
						kcdf = pd.DataFrame.from_dict(nx.average_neighbor_degree(g).items())
						kcdf['k'] = g.degree().values()
						dorig = pd.concat([dorig, kcdf])

				print "kron"
				gb = dorig.groupby(['k'])
				zz = len(gb[1].mean().values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(gb[1].mean().values), sa):
						print "(" + str(gb.mean().index[x]) + ", " + str(gb[1].mean().values[x]) + ")"


def draw_clustering_coefficients(orig_g_M, chunglu_M, HRG_M, pHRG_M, kron_M):

		if len(orig_g_M) is not 0:
				dorig = pd.DataFrame()
				for g in orig_g_M:
						degdf = pd.DataFrame.from_dict(g.degree().items())
						ccldf = pd.DataFrame.from_dict(nx.clustering(g).items())
						dat = np.array([degdf[0], degdf[1], ccldf[1]])
						df = pd.DataFrame(np.transpose(dat))
						df = df.astype(float)
						df.columns = ['v', 'k', 'cc']

						dorig = pd.concat([dorig, df])	# Appends to bottom new DFs

				print "orig"
				gb = dorig.groupby(['k'])
				zz = len(gb['cc'].mean().values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(gb['cc'].mean().values), sa):
						print "(" + str(gb['cc'].mean().index[x]) + ", " + str(gb['cc'].mean().values[x]) + ")"



		if len(chunglu_M) is not 0:
				dorig = pd.DataFrame()
				for g in chunglu_M:
						degdf = pd.DataFrame.from_dict(g.degree().items())
						ccldf = pd.DataFrame.from_dict(nx.clustering(g).items())
						dat = np.array([degdf[0], degdf[1], ccldf[1]])
						df = pd.DataFrame(np.transpose(dat))
						df = df.astype(float)
						df.columns = ['v', 'k', 'cc']

						dorig = pd.concat([dorig, df])	# Appends to bottom new DFs

				print "cl"
				gb = dorig.groupby(['k'])
				zz = len(gb['cc'].mean().values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(gb['cc'].mean().values), sa):
						print "(" + str(gb['cc'].mean().index[x]) + ", " + str(gb['cc'].mean().values[x]) + ")"

		if len(HRG_M) is not 0:
				dorig = pd.DataFrame()
				for g in HRG_M:
						degdf = pd.DataFrame.from_dict(g.degree().items())
						ccldf = pd.DataFrame.from_dict(nx.clustering(g).items())
						dat = np.array([degdf[0], degdf[1], ccldf[1]])
						df = pd.DataFrame(np.transpose(dat))
						df = df.astype(float)
						df.columns = ['v', 'k', 'cc']

						dorig = pd.concat([dorig, df])	# Appends to bottom new DFs

				print "hrg"
				gb = dorig.groupby(['k'])
				zz = len(gb['cc'].mean().values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(gb['cc'].mean().values), sa):
						print "(" + str(gb['cc'].mean().index[x]) + ", " + str(gb['cc'].mean().values[x]) + ")"

		if len(pHRG_M) is not 0:
				dorig = pd.DataFrame()
				for g in pHRG_M:
						degdf = pd.DataFrame.from_dict(g.degree().items())
						ccldf = pd.DataFrame.from_dict(nx.clustering(g).items())
						dat = np.array([degdf[0], degdf[1], ccldf[1]])
						df = pd.DataFrame(np.transpose(dat))
						df = df.astype(float)
						df.columns = ['v', 'k', 'cc']

						dorig = pd.concat([dorig, df])	# Appends to bottom new DFs

				print "phrgm"
				gb = dorig.groupby(['k'])
				zz = len(gb['cc'].mean().values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(gb['cc'].mean().values), sa):
						print "(" + str(gb['cc'].mean().index[x]) + ", " + str(gb['cc'].mean().values[x]) + ")"

		if len(kron_M) is not 0:
				dorig = pd.DataFrame()
				for g in kron_M:
						degdf = pd.DataFrame.from_dict(g.degree().items())
						ccldf = pd.DataFrame.from_dict(nx.clustering(g).items())
						dat = np.array([degdf[0], degdf[1], ccldf[1]])
						df = pd.DataFrame(np.transpose(dat))
						df = df.astype(float)
						df.columns = ['v', 'k', 'cc']

						dorig = pd.concat([dorig, df])	# Appends to bottom new DFs

				print "kron"
				gb = dorig.groupby(['k'])
				zz = len(gb['cc'].mean().values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(gb['cc'].mean().values), sa):
						print "(" + str(gb['cc'].mean().index[x]) + ", " + str(gb['cc'].mean().values[x]) + ")"


def draw_kcore_decomposition(orig_g_M, chunglu_M, HRG_M, pHRG_M, kron_M):
		dorig = pd.DataFrame()
		for g in orig_g_M:
				g.remove_edges_from(g.selfloop_edges())
				d = nx.core_number(g)
				df = pd.DataFrame.from_dict(d.items())
				df[[0]] = df[[0]].astype(int)
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		print "orig"

		if not dorig.empty :
				zz = len(dorig.mean(axis=1).values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(dorig.mean(axis=1).values), sa):
						print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"


		dorig = pd.DataFrame()
		for g in pHRG_M:
				d = nx.core_number(g)
				df = pd.DataFrame.from_dict(d.items())
				df[[0]] = df[[0]].astype(int)
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		print "phrg"
		if not dorig.empty :
				zz = len(dorig.mean(axis=1).values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(dorig.mean(axis=1).values), sa):
						print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"

		dorig = pd.DataFrame()
		for g in HRG_M:
				d = nx.core_number(g)
				df = pd.DataFrame.from_dict(d.items())
				df[[0]] = df[[0]].astype(int)
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		print "hrg"
		if not dorig.empty :
				zz = len(dorig.mean(axis=1).values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(dorig.mean(axis=1).values), sa):
						print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"

		dorig = pd.DataFrame()
		for g in chunglu_M:
				g.remove_edges_from(g.selfloop_edges())
				d = nx.core_number(g)
				df = pd.DataFrame.from_dict(d.items())
				df[[0]] = df[[0]].astype(int)
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		print "cl"
		if not dorig.empty :
				zz = len(dorig.mean(axis=1).values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(dorig.mean(axis=1).values), sa):
						print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"

		dorig = pd.DataFrame()
		for g in kron_M:
				d = nx.core_number(g)
				df = pd.DataFrame.from_dict(d.items())
				df[[0]] = df[[0]].astype(int)
				gb = df.groupby(by=[1])
				dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		print "kron"
		if not dorig.empty :
				zz = len(dorig.mean(axis=1).values)
				sa =	int(math.ceil(zz/75))
				if sa == 0: sa=1
				for x in range(0, len(dorig.mean(axis=1).values), sa):
						print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"


def external_rage(G,netname):
		import subprocess
		import networkx as nx
		from pandas import DataFrame
		from os.path import expanduser

		# giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
		giant_nodes = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)

		G = nx.subgraph(G, giant_nodes[0])
		tmp_file = "tmp_{}.txt".format(netname)
		with open(tmp_file, 'w') as tmp:
				for e in G.edges():
						tmp.write(str(int(e[0])+1) + ' ' + str(int(e[1])+1) + '\n')

		# args = ("wine", "./RAGE.exe", tmp_file)
		# args = ("/Volumes/Lexar/SeptBackupMBP/ToolSet/rage/Source_RAGE_unix/RAGE", tmp_file)
		# args = ("/data/cpennycu/rage/Source_RAGE_unix/RAGE", tmp_file)
		# args = ("/home/saguinag/Software/rage/Source_RAGE_unix/RAGE", tmp_file)
		if "Users" in expanduser('~').split('/'):
			args = ("/Users/saguinag/Research/rage/Source_RAGE_unix/RAGE", tmp_file)
		else:
			args = ("/home/saguinag/Software/rage/Source_RAGE_unix/RAGE", tmp_file)
		popen = subprocess.Popen(args, stdout=subprocess.PIPE)
		popen.wait()
		output = popen.stdout.read()

		# Results are hardcoded in the exe
		df = DataFrame.from_csv("./Results/UNDIR_RESULTS_tmp_{}.csv".format(netname), header=0, sep=',', index_col=0)
		df = df.drop('ASType', 1)
		return df


def tijana_eval_rgfd(G_df, H_df):
		T_G = 0.0
		T_H = 0.0
		RGFD = 0.0
		for column in G_df:
				T_G += sum(G_df[column][:])

		for column in H_df:
				T_H += sum(H_df[column][:])

		for column in G_df:
				N_G_i = sum(G_df[column][:])
				N_H_i = sum(H_df[column][:])
				if N_G_i == 0 or N_H_i == 0:
						print 0;
				RGFD += np.log10(N_G_i / T_G) - np.log10(N_H_i / T_H)

		return RGFD


def tijana_eval_compute_gcm(G_df):
		import scipy.stats

		l = len(G_df.columns)
		gcm = np.zeros((l, l))
		i = 0
		for column_G in G_df:
				j = 0
				for column_H in G_df:
						gcm[i, j] = scipy.stats.spearmanr(G_df[column_G].tolist(), G_df[column_H].tolist())[0]
						if scipy.isnan(gcm[i, j]):
								gcm[i, j] = 1.0
						j += 1
				i += 1
		return gcm


def tijana_eval_compute_gcd(gcm_g, gcm_h):
		import math

		if len(gcm_h) != len(gcm_g):
				raise "Graphs must be same size"
		s = 0
		for i in range(0, len(gcm_g)):
				for j in range(i, len(gcm_h)):
						s += math.pow((gcm_g[i, j] - gcm_h[i, j]), 2)

		gcd = math.sqrt(s)
		return gcd

def save_degree_probability_distribution(orig_g_M, chunglu_M, pHRG_M, kron_M,in_graph_str=''):
	from datetime import datetime
	dorig = pd.DataFrame()
	for g in orig_g_M:
			d = g.degree()
			df = pd.DataFrame.from_dict(d.items())
			gb = df.groupby(by=[1]).count()
			gb.columns=['cnt']
			gb['k']=gb.index
			print gb.head()
			dorig = pd.concat([dorig, gb], axis=1)	# Appends to bottom new DFs
	print "orig"
	if not dorig.empty :
		dorig['pk'] = dorig['cnt']/float(g.number_of_nodes())
		out_path = '../Results/orig_kdist_{}.tsv'.format(in_graph_str)
		dorig[['k','pk']].to_csv(out_path, sep='\t', index=False, header=True)

		# dorig = pd.DataFrame()
		# for g in HRG_M:
		#		 d = g.degree()
		#		 df = pd.DataFrame.from_dict(d.items())
		#		 gb = df.groupby(by=[1])
		#		 dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		# print "hrgm"
		# if not dorig.empty :
		#		 zz = len(dorig.mean(axis=1).values)
		#		 sa =	int(math.ceil(zz/75))
		#		 for x in range(0, len(dorig.mean(axis=1).values), sa):
		#				 print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"

	dorig = pd.DataFrame()
	for g in pHRG_M:
			d = g.degree()
			df = pd.DataFrame.from_dict(d.items())
			gb = df.groupby(by=[1]).count()
			gb.columns=['cnt']
			# gb['k']=gb.index
			dorig = pd.concat([dorig, gb], axis=1)	# Appends to bottom new DFs
	print "phrgm"
	if not dorig.empty :
		"""
			zz = len(dorig.mean(axis=1).values)
			sa =	int(math.ceil(zz/75))
			for x in range(0, len(dorig.mean(axis=1).values), sa):
					print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"
		"""
		dorig['pk'] = dorig.mean(axis=1)/float(g.number_of_nodes())
		dorig['k'] = dorig.index # print dorig.head()

		out_path = '../Results/phrg_kdist_{}.tsv'.format(in_graph_str)
		dorig[['k','pk']].to_csv(out_path, sep='\t', index=False, header=True)

	dorig = pd.DataFrame()
	for g in chunglu_M:
			d = g.degree()
			df = pd.DataFrame.from_dict(d.items())
			gb = df.groupby(by=[1])
			gb.columns=['cnt']
			dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
	print "cl"
	if not dorig.empty :
		"""
		zz = len(dorig.mean(axis=1).values)
		sa =	int(math.ceil(zz/75))
		for x in range(0, len(dorig.mean(axis=1).values), sa):
				print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"
		"""
		dorig['pk'] = dorig.mean(axis=1)/float(g.number_of_nodes())
		dorig['k'] = dorig.index # print dorig.head()

		out_path = '../Results/clgm_kdist_{}.tsv'.format(in_graph_str)
		dorig[['k','pk']].to_csv(out_path, sep='\t', index=False, header=True)

	dorig = pd.DataFrame()
	for g in kron_M:
			d = g.degree()
			df = pd.DataFrame.from_dict(d.items())
			gb = df.groupby(by=[1])
			gb.columns=['cnt']
			dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
	print "kron"
	if not dorig.empty :
		"""
			zz = len(dorig.mean(axis=1).values)
			sa =	int(math.ceil(zz/75))
			for x in range(0, len(dorig.mean(axis=1).values), sa):
					print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"
		"""
		dorig['pk'] = dorig.mean(axis=1)/float(g.number_of_nodes())
		dorig['k']	= dorig.index # print dorig.head()

		out_path = '../Results/kpgm_kdist_{}.tsv'.format(in_graph_str)#str(datetime.now()).replace(' ','_'))
		dorig[['k','pk']].to_csv(out_path, sep='\t', index=False, header=True)

#save_eigenvector_centrality
def save_eigenvector_centrality(orig_g_M, chunglu_M, pHRG_M, kron_M,in_graph_str=''):
	#from datetime import datetime
	dorig = pd.DataFrame()
	for g in orig_g_M:
			d = g.degree()
			df = pd.DataFrame.from_dict(d.items())
			gb = df.groupby(by=[1]).count()
			gb.columns=['cnt']
			gb['k']=gb.index
			print gb.head()
			dorig = pd.concat([dorig, gb], axis=1)	# Appends to bottom new DFs
	print "orig"
	if not dorig.empty :
		dorig['pk'] = dorig['cnt']/float(g.number_of_nodes())
		out_path = '../Results/orig_kdist_{}.tsv'.format(in_graph_str)
		dorig[['k','pk']].to_csv(out_path, sep='\t', index=False, header=True)

		# dorig = pd.DataFrame()
		# for g in HRG_M:
		#		 d = g.degree()
		#		 df = pd.DataFrame.from_dict(d.items())
		#		 gb = df.groupby(by=[1])
		#		 dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
		# print "hrgm"
		# if not dorig.empty :
		#		 zz = len(dorig.mean(axis=1).values)
		#		 sa =	int(math.ceil(zz/75))
		#		 for x in range(0, len(dorig.mean(axis=1).values), sa):
		#				 print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"

	dorig = pd.DataFrame()
	for g in pHRG_M:
			d = g.degree()
			df = pd.DataFrame.from_dict(d.items())
			gb = df.groupby(by=[1]).count()
			gb.columns=['cnt']
			# gb['k']=gb.index
			dorig = pd.concat([dorig, gb], axis=1)	# Appends to bottom new DFs
	print "phrgm"
	if not dorig.empty :
		"""
			zz = len(dorig.mean(axis=1).values)
			sa =	int(math.ceil(zz/75))
			for x in range(0, len(dorig.mean(axis=1).values), sa):
					print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"
		"""
		dorig['pk'] = dorig.mean(axis=1)/float(g.number_of_nodes())
		dorig['k'] = dorig.index # print dorig.head()

		out_path = '../Results/phrg_kdist_{}.tsv'.format(in_graph_str)
		dorig[['k','pk']].to_csv(out_path, sep='\t', index=False, header=True)

	dorig = pd.DataFrame()
	for g in chunglu_M:
			d = g.degree()
			df = pd.DataFrame.from_dict(d.items())
			gb = df.groupby(by=[1])
			gb.columns=['cnt']
			dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
	print "cl"
	if not dorig.empty :
		"""
		zz = len(dorig.mean(axis=1).values)
		sa =	int(math.ceil(zz/75))
		for x in range(0, len(dorig.mean(axis=1).values), sa):
				print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"
		"""
		dorig['pk'] = dorig.mean(axis=1)/float(g.number_of_nodes())
		dorig['k'] = dorig.index # print dorig.head()

		out_path = '../Results/clgm_kdist_{}.tsv'.format(in_graph_str)
		dorig[['k','pk']].to_csv(out_path, sep='\t', index=False, header=True)

	dorig = pd.DataFrame()
	for g in kron_M:
			d = g.degree()
			df = pd.DataFrame.from_dict(d.items())
			gb = df.groupby(by=[1])
			gb.columns=['cnt']
			dorig = pd.concat([dorig, gb.count()], axis=1)	# Appends to bottom new DFs
	print "kron"
	if not dorig.empty :
		"""
			zz = len(dorig.mean(axis=1).values)
			sa =	int(math.ceil(zz/75))
			for x in range(0, len(dorig.mean(axis=1).values), sa):
					print "(" + str(dorig.mean(axis=1).index[x]) + ", " + str(dorig.mean(axis=1).values[x]) + ")"
		"""
		dorig['pk'] = dorig.mean(axis=1)/float(g.number_of_nodes())
		dorig['k']	= dorig.index # print dorig.head()

		out_path = '../Results/kpgm_kdist_{}.tsv'.format(in_graph_str)#str(datetime.now()).replace(' ','_'))
		dorig[['k','pk']].to_csv(out_path, sep='\t', index=False, header=True)
