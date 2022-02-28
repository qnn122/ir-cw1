from matplotlib.pyplot import text
import spacy
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime 
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------- TERM FREQUENCY ------------
def run_term_frequency(filename, issave=False, verbose=False):
	"""Run experiment of producing term frequency given a collection of documents
	Returned files:
		vocab.txt
		freq.json
	"""
	# Loading data and prepare
	print('### Loading data ###')
	with open(filename, 'r') as f:
		passages = f.readlines()

	print('### Processing ###')
	# MAIN PROCESSING
	vocab, freq = text_stats(passages)

	# Calculate elapsed time and printe result
	if verbose:
		print(f'vocab: \n {vocab}: \n\n')
		print(f'freq: \n {freq}\n\n')
		print('Vocab size: ', len(vocab))

	# save
	if issave:
		import simplejson
		with open('vocab.txt', 'w') as f:
			simplejson.dump(vocab, f)

		import json
		with open('freq.json', 'w') as f:
			json.dump(freq, f)
   
def text_stats(passages: List[str]):
	"""Produce vocabulary of a list of documents and term frequencies

	Args:
		passages (List[str]): list of passages

	Returns:
		[type]: [description]
	"""
	vocab = []
	freq = {}
	nlp = spacy.blank('en')
	for passage in tqdm(passages):
		doc = nlp(passage)
		for term in doc:
			if str(term) not in vocab:
				vocab.append(str(term))
				freq[str(term)] = 1
			else:
				freq[str(term)] += 1
	return vocab, freq

# ---------- ZIPF's LAW ------------
def run_zipf_law(T=1, isplot=False, issave=False):
	"""Plot and analyze term frequency.

	Args:
		T (int, optional): Threshold of frequency below which a term is removed. Defaults to 1.
		isplot (bool, optional): Whether to show figures. Defaults to False.
	"""
	freqfile = 'freq.json'	

	# Convert term frequency to dataframe for easier manipulation (rank, normalisation) and visualization
	df_filtered = process_terms(freqfile, T)
	df_filtered = df_filtered.reset_index()
	N = len(df_filtered)
	print(f'Number of terms: {N}')

	# create a column for rank since index has been altered by the filtering
	df_filtered['_rank'] = np.array(pd.RangeIndex(1, len(df_filtered.index)+1))
	S = df_filtered['freq'].sum()

	# calculate zip score
	df_filtered['freq_norm'] = df_filtered['freq']/S	# normalized frequency
	df_filtered['zipf'] = None # necessary
	#df_filtered['zipf'] = df_filtered.apply(lambda row: zipf(float(row._rank), 1, N), axis=1)
	df_filtered['zipf'] = pd.Series(zipf_np(np.array(df_filtered._rank, dtype=float), 1, N)).reindex(df_filtered.index)
 
	# check if rank*freq_norm converts to constant
	df_filtered['rank*freq_norm'] = df_filtered['_rank']*df_filtered['freq_norm']

	# --------------- PLOTTING --------------
	# plot zipf score (theoritical) and actual term frequency (empirical)
	fig, axes = plt.subplots(ncols=2, figsize=(10, 5))

	# Plot term frequency (normalised)
	#df_filtered.plot(x='_rank', y=['freq_norm'], ax=axes[0])
	df_filtered.plot.bar(x='_rank', y='freq_norm', rot=90, ax=axes[0])
	axes[0].set_xlabel('rank'), axes[0].set_ylabel('normalised freq.'), axes[0].get_legend().remove()
	axes[0].set_xticklabels([]), axes[0].set_xticks([])

	# zipf
	df_filtered.plot(x='_rank', y=['freq_norm', 'zipf'], loglog=True, ax=axes[1])
	axes[1].set_xlabel('rank'), axes[1].legend(labels=['normalised freq.',"zipf's law"])

	# (optional) see if rank*freq_norm converts
	df_filtered.plot(x='_rank', y=['rank*freq_norm'])
 
	if isplot:
		plt.show()
  
	if issave:
    	#extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
		fig.savefig(f'task1_termfreq_{T}_{N}.png')

	# check if the zipf law holds for this dataset
	zipf_log = np.log(df_filtered['zipf'])
	freq_norm_log = np.log(df_filtered['freq_norm'])
	print('Correlation (log scale):    {:.4f}'.format(calc_correlation(zipf_log, freq_norm_log)))	# correlation in log domain
	print('Correlation (normal scale): {:.4f}'.format(calc_correlation(df_filtered['freq_norm'], df_filtered['zipf'])))	# in normal domain
 
def process_terms(freqfile, T=1, stopword=None):
	with open(freqfile, 'r') as f:
		freq = json.load(f)

	# convert to df for ranking and visualization
	freq_items = freq.items()
	freq_list = list(freq_items)

	df = pd.DataFrame(freq_list)
	df = df.rename(columns={0:'term', 1:'freq'})

	# rank term based on frequency
	df_ranked = df.sort_values(by=['freq'], ascending=False, ignore_index=True)

	# Filtered out term whose occurences are fewer than certain number (easier visualization)
	df_filtered = df_ranked.drop(df[df.freq < T].index)
	return df_filtered

def zipf(k, s, N):
	"""Calcular zipf score
	"""
	deno = 0
	for i in range(1, N+1):
		deno += i**(-s)
	return k**(-s)/deno

def zipf_np(ks, s, N):
	"""Calculate zipf score of a numpy vector
 
	ks : float type numpy vector of ranks
	"""
	deno = 0
	for i in range(1, N+1):
		deno += i**(-s)
	return ks**(-s)/deno

def plot_zipf(k_end, s, N, loglog=False):
	ks = []
	fs = []
	for k in range(1, k_end+1):
		ks.append(k)
		fs.append(zipf(k, s, N))
	if loglog:
		plt.loglog(np.array(ks), np.array(fs))
	else:
		plt.plot(np.array(ks), np.array(fs))

def calc_correlation(x_values, y_values):
	correlation_matrix = np.corrcoef(x_values, y_values)
	correlation_xy = correlation_matrix[0,1]
	r_squared = correlation_xy**2
	return r_squared

 
if __name__ == '__main__':
	start_time = datetime.now() 
	#run_term_frequency('coursework-1-data/passage-collection_10.txt', ismulti=True, verbose=True)
	run_zipf_law(T=100, isplot=False, issave=True)
	time_elapsed = datetime.now() - start_time 	
	print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))