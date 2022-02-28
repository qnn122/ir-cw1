from datetime import datetime 
from math import log
import copy

from task3 import RetrieverBase

# input file
datafile = 'coursework-1-data/candidate-passages-top1000.tsv'
test_queries = 'coursework-1-data/test-queries.tsv'
idf_file = '.cache/idf.json'
tfidf_file = '.cache/tfidf.json'
freq_file = '.cache/freq.json'
dfq_file = '.cache/dfq.json'

class RetrieverProb(RetrieverBase):
	def __init__(self, datafile='coursework-1-data/candidate-passages-top1000.tsv', 
					test_queries='coursework-1-data/test-queries.tsv', 
					idf_file='.cache/idf.json', 
					tfidf_file='.cache/idf.json',
					freq_file='.cache/freq.json',
					dfq_file = '.cache/dfq.json') -> None:

		super().__init__(datafile, test_queries, idf_file, 
                   		tfidf_file, freq_file, dfq_file)

	def _load_essentials(self, term, pid):
		tf = copy.deepcopy(self.idf[term][str(pid)])
		D = self.calc_dl(self.get_passage(int(pid))) 	# doc length |D|
		V = len(self.vocab)								# vocab length |V|
		return tf, D, V

	def laplace(self, term, pid):
		tf, D, V = self._load_essentials(term, pid)
		prob = (tf + 1) / (D + V)
		return log(prob)

	def lindstone(self, term, pid):
		tf, D, V = self._load_essentials(term, pid)
		eps = 0.1
		prob = (tf + eps) / (D + eps*V)
		return log(prob)

	def dirichlet(self, term, pid):
		tf, D, V = self._load_essentials(term, pid)
		mu = 5000
		alpha = D / (D + mu)
		doc_term = tf / D
		col_term = self.freq[term] / self.T
		prob = alpha*doc_term + (1 - alpha)*col_term
		return log(prob)

	def base(self, term, pid):
		tf, D, V = self._load_essentials(term, pid)
		prob = prob = tf / D
		return log(prob)


def query_single(qid, smoothing):
	query = retriever.get_query(qid)
	ans = retriever.qllm(query, smoothing=smoothing, cutoff=5, verbose=True)

	print('\n------------------------')
	print(f'Query: {query}')
	print('------------------------\n')
	print(f'Anwers: ({smoothing})')
	for i, doc in enumerate(ans):
		pid, score = doc
		print(f'  {i+1} | {retriever.get_passage(int(pid))} ({round(score,3)})\n')
  
  
if __name__ == '__main__':
    # Initializing
	start_time = datetime.now() 
	retriever = RetrieverProb()
	retriever.stats()
 
	# main processing
	retriever.run_exp(test_queries_file='coursework-1-data/test-queries.tsv', 
                    model='laplace', cutoff=5)
 
	retriever.run_exp(test_queries_file='coursework-1-data/test-queries.tsv', 
                    model='lindstone', cutoff=5)
 
	# Measure and display elapsed time
	time_elapsed = datetime.now() - start_time 	
	print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
 
'''
retriever = RetrieverProb()
retriever.stats()

# 
qid = 1108939
#query_single(qid, smoothing='no smoothing')
#query_single(qid, smoothing='laplace')
#query_single(qid, smoothing='lindstone')
query_single(qid, smoothing='dirichlet')
'''

'''
qid = 1108939
what slows down the flow of blood
['slows', 'flow', 'blood]
'''