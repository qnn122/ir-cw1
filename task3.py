import os
import srsly
from task2 import make_idf
import pandas as pd
import spacy
from tqdm import tqdm
from datetime import datetime 
from math import log
import string
import copy
import concurrent
import csv
import numpy as np
from numpy import dot
from numpy.linalg import norm
from collections import Counter

# input file
datafile = 'coursework-1-data/candidate-passages-top1000.tsv'
test_queries = 'coursework-1-data/test-queries.tsv'
idf_file = '.cache/idf.json'
tfidf_file = '.cache/tfidf.json'
freq_file = '.cache/freq.json'
dfq_file = '.cache/dfq.json'


# ------------------- UTILITIES -------------------
def load_nlp():
	"""Load tokenizer. Inherited from spacy blank english tokenizer.
	Add `\` and `.` to seperate terms of hyperlinks
	"""
	nlp = spacy.blank('en')
	nlp.tokenizer.url_match = None
	infixes = nlp.Defaults.infixes + [r'\.']
	infix_regex = spacy.util.compile_infix_regex(infixes)
	nlp.tokenizer.infix_finditer = infix_regex.finditer
	return nlp

def load_removed():
	"""Load characters/words to be removed. 
	Including: spacy's common stopwords and punctuation marks
	"""
	en = spacy.load('en_core_web_sm')
	stopwords = list(en.Defaults.stop_words)
	stopwords_cap = [word.capitalize() for word in stopwords]
	marks = list(string.punctuation)
	removed = stopwords + stopwords_cap + marks
	return removed

def idf_filter(idf, terms):
    """idf with only selected `terms`
    
    {
		'term1': {'pid1': score, 'pid2': score, ...}
		'term2': {'pid1': score, 'pid2': score, ...}
		...
	}
    """
    return dict([ (i,idf[i]) for i in idf if i in set(terms) ])


# Load util. functions
nlp = load_nlp()
removed = load_removed()
#dict_filter = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])


# ------------------- MAIN CLASS -------------------
class RetrieverBase:
	def __init__(self, datafile='coursework-1-data/candidate-passages-top1000.tsv', 
              		   test_queries='coursework-1-data/test-queries.tsv', 
                   	   idf_file='.cache/idf.json', 
                       tfidf_file='.cache/idf.json',
                       freq_file='.cache/freq.json',
                       dfq_file = '.cache/dfq.json') -> None:
		# preprare datafile (content of queries and passages)
		df = pd.read_csv(datafile, delimiter='\t', header=None)
		df = df.rename(columns={0:'qid', 1:'pid', 2:'query', 3:'passage'})
		self.df = df
  
		# prepare test queries dataframe 
		test_queries = pd.read_csv(test_queries, delimiter='\t')
		test_queries = test_queries.rename(columns={0:'qid', 1:'query'})
		self.test_queries = test_queries
  
		# prepare representation dataframe
		if os.path.exists(idf_file):
			idf = srsly.read_json(idf_file); freq = srsly.read_json(freq_file)
		else:
			idf, freq, _ = make_idf(datafile)
		self.idf = idf
		
		'''
		if os.path.exists(tfidf_file):
			tfidf = srsly.read_json(tfidf_file); freq = srsly.read_json(freq_file)
		else:
			tfidf, freq, _ = make_idf(datafile)
		self.tfidf = tfidf
		'''
  
		if os.path.exists(dfq_file):
			dfq = srsly.read_json(dfq_file); freq = srsly.read_json(freq_file)
		else:
			_, freq, dfq = make_idf(datafile)
		self.dfq = dfq
  
		# Other handy variables
		self.qids = list(set(list(df['qid'])))
		self.pids = list(set(list(df['pid'])))
		self.freq = freq
		self.vocab = list(freq.keys())	# idf can be used here but freq provide the ranking
		self.N = len(self.pids)			# number of docs (passages)
		self.T = sum(self.freq.values())	# no. of all tokens in the collection
		self.avdl = self.get_avdl()
   
   # ------------ UTILITIES ------------
	def get_passage(self, pid: int):
		"""Get the actual passage (string) given its id `pid`
  		"""
		return list(self.df[self.df['pid']==pid]['passage'])[0]

	def get_query(self, qid):
		"""Get the actual query (string) given its id `qid`
  		"""
		return list(self.df[self.df['qid']==qid]['query'])[0]

	def get_avdl(self):
		"""Return the average document length (of the collection)
  		"""
		total_docs = len(self.pids)
		len_all_docs = 0
		for pid in list(self.dfq.keys()):
			len_all_docs += sum(self.dfq[pid].values())
		return len_all_docs/total_docs
		
	def calc_dl(self, doc):
		"""Return document length (number of tokens)
  		"""
		return len(nlp(doc))

	def calc_u(self, terms):
		"""Return the union set of pids of the given term set
  		"""
		u = set()	# union of terms' pids
		for term in terms:
			pids = set(list(self.idf[term]))
			u = u.union(pids)
		return u

	def stats(self):
		"""Display some key statistics of the data
  		"""
		print(f'Vocab size: {len(self.vocab)}')
		print(f'No. of unique passages: {self.N}')
		print(f'No. of unique queries:  {len(self.qids)}')
		print(f'No. of passages/query:  {self.N/len(self.qids)}')
		print(f'avdl: {self.avdl}')

	def run_exp(self, test_queries_file, model, cutoff=5, verbose=False):
		# initialize output
		outputfile = model + '.csv'
		with open(outputfile, 'w') as output: 
			writer = csv.writer(output, delimiter=',') 
			writer.writerow(['qid', 'pid', 'score'])

		# processing
		with open(test_queries_file, 'r') as csvfile: 
			reader = csv.reader(csvfile, delimiter='\t') 
			reader = list(reader)
			for row in tqdm(reader):
				qid, query = row
				results = self.retrieve(query, model, cutoff=cutoff, verbose=verbose)
				with open(outputfile, 'a') as output:
					writer = csv.writer(output, delimiter=',') 
					for result in results:
						pid, score = result
						writer.writerow([qid, pid, score])
		
	def retrieve(self, query, model, cutoff=5, verbose=False):
		"""Calculate bm25 score of the entire doc collection given a query (string)
		Return top `cutoff` documents with highest bm25 score

		Output: list of tuple (pid, score)
			Ex: [('7651010', 16.95925678633877), ('5291621', 16.219478389024623),...]
		"""
		model = eval(f'self.{model}')
		# Calculate  score
		#terms = [str(tok) for tok in nlp(query) if str(tok) not in removed]  
		terms = []
		for tok in nlp(query):
			tok = str(tok).lower()
			if tok not in self.vocab and tok not in removed:
				tqdm.write(f'New word: {tok}')
				continue
			elif tok in removed:
				continue
			else:
				terms.append(tok)

		score = idf_filter(self.idf, terms)

		q_vec = np.zeros((len(terms)))
		freq_term_query = Counter(terms)
		for i, term in enumerate(tqdm(terms, disable=not(verbose))):
			n_t = len(list(self.idf[term]))
			idf_term = log(self.N / n_t)
			q_vec[i] = idf_term * freq_term_query[term]/len(terms)
			for pid in tqdm(list(self.idf[term]), disable=not(verbose)):
				score[term][pid] = model(term, int(pid))

		u = list(self.calc_u(terms)) # union of docs terms in query involves
		output = dict(zip(u, [0]*len(u)))  
		freq_term_query = Counter(terms)
		for pid in u:
			if model.__name__ == 'tfidf':
				d_vec = np.zeros((len(terms)))
				#q_vec = np.zeros((len(terms)))
				for i, term in enumerate(terms):
					if pid in score[term]:
						n_t = len(list(self.idf[term]))
						idf_term = log(self.N / n_t)
						d_vec[i] = score[term][pid] # tfidf of doc
						#q_vec[i] = idf_term * freq_term_query[term]/len(terms) # tfidf of query
				cos_sim = dot(q_vec, d_vec)/(norm(q_vec)*norm(d_vec))
				output[pid] = cos_sim # cosine similarity
			else:
				for term in terms:
					if pid in score[term]:
						output[pid] += score[term][pid]

		output_sorted = [(k, v) for k, v in sorted(output.items(), key=lambda item: (item[1], len(self.get_passage(int(item[0])))), reverse=True)]
		if len(output_sorted) < 101:
			print(f'{model.__name__} | {query}')
		return output_sorted[:cutoff]

	def bm25(self, term, pid):
		"""Calculate bm25 score of a document given a (single) term
		"""
		k1, k2, b = 1.2, 100, 0.75
		n = len(self.idf[term])
		dl = self.calc_dl(self.get_passage(int(pid)))
		if str(pid) in self.idf[term]:
			f = self.idf[term][str(pid)]
		else:
			f = 0
		qf = 1
		r, R = 0, 0

		K = k1*((1-b) + b*dl/self.avdl)
		log_term = log(( (r+0.5)*(self.N-n-R+r+0.5) ) / ( (n-r+0.5)*(R-r+0.5) ))
		sec_term = ((k1 + 1)*f/(K+f)) * ((k2+1)*qf/(k2+qf))
		return log_term * sec_term

	def tfidf(self, term, pid):
		"""Calculate TF-IDF score
  		"""
		n_t = len(list(self.idf[term]))
		idf_term = log(self.N / n_t)
		return self.idf[term][str(pid)] * idf_term


def query_single(qid, model):
	query = retriever.get_query(qid)
	ans = retriever.retrieve(query, model, cutoff=10, verbose=True)

	print('\n------------------------')
	print(f'Query: {query}')
	print('------------------------\n')
	print('Anwers:')
	for i, doc in enumerate(ans):
		pid, score = doc
		print(f'  {i+1} | {retriever.get_passage(int(pid))} ({round(score,3)})\n')
  
  
if __name__ == '__main__':
	print('---------- TASK 3 ----------')
	# Initializing
	start_time = datetime.now() 
	retriever = RetrieverBase()
	retriever.stats()

	# main processing
	c = 5
	retriever.run_exp(test_queries_file='coursework-1-data/test-queries.tsv', 
					model='tfidf', cutoff=c)

	retriever.run_exp(test_queries_file='coursework-1-data/test-queries.tsv', 
					model='bm25', cutoff=c)

	# Measure and display elapsed time
	time_elapsed = datetime.now() - start_time 	
	print('Time elapsed (hh:mm:ss.ms) {}\n'.format(time_elapsed))