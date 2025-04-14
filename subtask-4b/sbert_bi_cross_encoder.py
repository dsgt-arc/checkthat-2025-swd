import subprocess
subprocess.check_call(["python", "-m", "pip", "install", "numpy"])
subprocess.check_call(["python", "-m", "pip", "install", "pandas"])
subprocess.check_call(["python", "-m", "pip", "install", "sentence-transformers"])

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sentence_transformers import CrossEncoder

PATH_COLLECTION_DATA = 'subtask4b_collection_data.pkl'
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)

PATH_QUERY_DATA = 'subtask4b_query_tweets_dev.tsv'
df_query = pd.read_csv(PATH_QUERY_DATA, sep = '\t')

# Retrieval
bi_encoder = SentenceTransformer('msmarco-distilbert-base-v4')

paper_texts = df_collection['title'] + " " + df_collection['abstract']
corpus = df_collection[:][['title', 'abstract']].apply(lambda x: f"{x['title']} {x['abstract']}", axis=1).tolist()
cord_uids = df_collection['cord_uid'].tolist()
paper_embeddings = bi_encoder.encode(paper_texts.tolist(), convert_to_tensor=True, show_progress_bar=True)
tweet_embeddings = bi_encoder.encode(df_query['tweet_text'].tolist(), convert_to_tensor=True, show_progress_bar=True)

def retrieve_with_bi_encoder(tweet_embeddings, paper_embeddings, cord_uids, top_k=32):
    retrieved_docs_list = []
    retrieved_scores_list = []

    for tweet_embedding in tweet_embeddings:
        hits = util.semantic_search(tweet_embedding, paper_embeddings, top_k=top_k)[0]
        sorted_docs = [cord_uids[hit['corpus_id']] for hit in hits]
        sorted_scores = [hit['score'] for hit in hits]

        retrieved_docs_list.append(sorted_docs)
        retrieved_scores_list.append(sorted_scores)

    return retrieved_docs_list, retrieved_scores_list

top_k = 100
df_query['bi_encoder_retrieved'], df_query['bi_encoder_scores'] = retrieve_with_bi_encoder(
    tweet_embeddings, paper_embeddings, cord_uids, top_k=top_k)

# Evaluate retrieved candidates using MRR@k
def get_performance_mrr(data, col_gold, col_pred, list_k = [1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) if x[col_gold] in [i for i in x[col_pred][:k]] else 0), axis=1)
        #performances.append(data["in_topx"].mean())
        d_performance[k] = data["in_topx"].mean()
    return d_performance

# MRR@k for bi-encoder
results = get_performance_mrr(df_query, 'cord_uid', 'bi_encoder_retrieved')
print(results)

# Rerank
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(df, cross_encoder):
    reranked_docs_list = []
    reranked_scores_list = []

    for idx, query in enumerate(df['tweet_text'].tolist()):
        # Get retrieved documents from Bi-Encoder
        retrieved_docs = df.iloc[idx]['bi_encoder_retrieved']
        retrieved_texts = [corpus[cord_uids.index(doc)] for doc in retrieved_docs]

        # Rank retrieved documents using Cross-Encoder
        ranked_results = cross_encoder.rank(
            query=query,
            documents=retrieved_texts,
            top_k=len(retrieved_texts), 
            show_progress_bar=True
        )

        # Extract ranked document IDs and scores
        sorted_docs = [retrieved_docs[result['corpus_id']] for result in ranked_results]
        sorted_scores = [result['score'] for result in ranked_results]

        reranked_docs_list.append(sorted_docs)
        reranked_scores_list.append(sorted_scores)

    return reranked_docs_list, reranked_scores_list

df_query['cross_encoder_reranked'], df_query['cross_encoder_scores'] = rerank_with_cross_encoder(df_query, cross_encoder)

#MRR@k for cross-encoder
results = get_performance_mrr(df_query, 'cord_uid', 'cross_encoder_reranked')
print(results)


