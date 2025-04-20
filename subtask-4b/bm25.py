import numpy as np
from rank_bm25 import BM25Okapi

class BM25:
    def __init__(self, corpus, cord_uids):
        self.corpus = corpus
        self.cord_uids = cord_uids
        self.bm25 = BM25Okapi(self.tokenize_corpus())
        self.text2bm25top = {}

    def get_top_cord_uids(self, query, k=100):
        query = str(query).replace(",", "")
        if query in self.text2bm25top.keys():
            return self.text2bm25top[query]
        else:
            tokenized_query = query.split(' ')
            doc_scores = self.bm25.get_scores(tokenized_query)
            indices = np.argsort(-doc_scores)[:k]
            bm25_topk = [self.cord_uids[x] for x in indices]
            self.text2bm25top[query] = bm25_topk
            return bm25_topk
        
    def retrieve_paper(paper_ids, df_collection):
        paper_dict = {}
        for id in paper_ids:
            paper_data = df_collection[df_collection['cord_uid'] == id]
            title = paper_data.iloc[0]['title']
            abstract = paper_data.iloc[0]['abstract']
            paper_dict[id] = {'title': title, 'abstract': abstract}
        return paper_dict

    def tokenize_corpus(self):
        return [doc.split(' ') for doc in self.corpus]
