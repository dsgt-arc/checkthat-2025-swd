import numpy as np
from bm25_pt import BM25
from torch import Tensor

class BM25_Pytorch:
    def __init__(self, corpus, cord_uids, device=None):
        self.corpus = corpus
        self.cord_uids = cord_uids
        self.bm25 = BM25(device=device)
        self.bm25.index(self.corpus)
        
    def get_top_cord_uids(self, query, k=100):
        doc_scores = Tensor.cpu(self.bm25.score(query))
        indices = np.argsort(-doc_scores)[:k]
        bm25_topk = [self.cord_uids[x] for x in indices]
        return bm25_topk
        
    def retrieve_paper(paper_ids, df_collection):
        paper_dict = {}
        for id in paper_ids:
            paper_data = df_collection[df_collection['cord_uid'] == id]
            title = paper_data.iloc[0]['title']
            abstract = paper_data.iloc[0]['abstract']
            paper_dict[id] = {'title': title, 'abstract': abstract}
        return paper_dict

   
