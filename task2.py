import pandas as pd
import numpy as np
import spacy
from datetime import datetime 
from tqdm import tqdm
from collections import Counter
import srsly
import string 


nlp = spacy.blank('en')
nlp.tokenizer.url_match = None
infixes = nlp.Defaults.infixes + [r'\.']
infix_regex = spacy.util.compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_regex.finditer

en = spacy.load('en_core_web_sm')
stopwords = list(en.Defaults.stop_words)
stopwords_cap = [word.capitalize() for word in stopwords]
marks = list(string.punctuation)
    
removed = stopwords + stopwords_cap + marks

# -----------------
# load passages/docs
def make_idf(datafile, issave=False):
	datafile = 'coursework-1-data/candidate-passages-top1000.tsv' # might contain duplicate docs but has pid
	df = pd.read_csv(datafile, delimiter='\t', header=None)
	df = df.rename(columns={0:'qid', 1:'pid', 2:'query', 3:'passage'})
	pids = list(set(list(df['pid'])))	# create set of unique passages
	passages = list(set(list(df['passage'])))	# create set of unique passages

	freqfile = 'freq.json'
	term_freq = srsly.read_json(freqfile)
	vocab = list(term_freq.keys())
	dfq = {}
	idf = {}
	for pid in tqdm(pids): 
		passage = df.loc[df['pid']==pid, 'passage'].values[0]
		tokens = [str(token) for token in nlp(passage) if str(token) not in removed]
		freq = Counter(tokens)
		dfq[pid] = freq

		for term in freq.keys():
			if term not in idf:
				idf[term] = {}
			if pid in idf[term]:
				idf[term][pid] += freq[term]
			else:
				idf[term][pid] = freq[term]

	vocab = list(idf.keys()) # better vocab, without '/n', ' /n', '  /n'
	freq = {}
	for term in vocab:
		#term = 'the'
		freq_term = 0
		for doc in idf[term]:
			freq_term += idf[term][doc]
		freq[term] = freq_term
	
	freq_sorted = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}

	if issave:
		srsly.write_json('.cache/idf.json', idf)
		srsly.write_json('.cache/freq.json', freq_sorted)
		srsly.write_json('.cache/dfq.json', dfq)
 
	return idf, freq_sorted, dfq


if __name__ == '__main__':
	datafile = 'coursework-1-data/candidate-passages-top1000.tsv'
	make_idf(datafile, issave=True)