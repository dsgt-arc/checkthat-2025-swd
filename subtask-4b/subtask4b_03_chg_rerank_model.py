import argparse
import numpy as np
import pandas as pd
import time
import torch

from bm25_pytorch import BM25_Pytorch
from rerank import Rerank
from util import get_performance_mrr, retrieve_paper, output_file, preprocess

EXPERIMENT = "03_chg_rerank_model"


def retrieve(df_collection, df_query, f, device=None, mrr_k = [1, 5, 10]):
  # Retrieval - Create the BM25 corpus (baseline)
  corpus = df_collection[:][['title', 'abstract']].apply(lambda x: f"{x['title']} {x['abstract']}", axis=1).tolist()
  cord_uids = df_collection[:]['cord_uid'].tolist()

  bm25 = BM25_Pytorch(corpus=corpus, cord_uids=cord_uids, device=device)
    
  # Retrieve topk candidates using the BM25 model
  df_query['bm25_topk'] = df_query['tweet_text'].apply(lambda x: bm25.get_top_cord_uids(x))

  results = get_performance_mrr(df_query, 
                                col_gold='cord_uid', 
                                col_pred='bm25_topk', 
                                list_k = mrr_k)
  
  f.write("EXPERIMENT {} BASELINE (BM25) RESULTS:\n".format(EXPERIMENT))
  f.write(str(results))
  f.write("\n\n")
  
  return df_query
  
  
def rerank(df_collection, df_query, f, rerank_model, rerank_k, device=None, mrr_k = [1, 5, 10]):
  rerank = Rerank(model_name=rerank_model, device=device)
      
  df_query['title_abstract'] = df_query['bm25_topk'].apply(lambda row: retrieve_paper(df_collection=df_collection, paper_ids=row))
  df_query['bm25_cross_encoder_topk'] = df_query.apply(lambda row: rerank.rerank_with_crossencoder(row, k=rerank_k), axis=1)

  # Check the result (this will contain the tweet and paper pairs)
  results = get_performance_mrr(df_query, 
                                col_gold='cord_uid', 
                                col_pred='bm25_cross_encoder_topk', 
                                list_k = mrr_k)
  
  f.write("EXPERIMENT {} RERANK RESULTS:\n".format(EXPERIMENT))
  f.write(str(results))
  f.write("\n")
  
  return df_query
 
 
def main(path_collection_data, path_query_data, output_dir, rerank_model, rerank_k, list_mrr_k):
  
  df_collection = pd.read_pickle(path_collection_data)
  df_collection = preprocess(df=df_collection)
  df_query = pd.read_csv(path_query_data, sep = '\t')
  df_query = preprocess(df=df_query)
  
  device = torch.device("cuda" if torch.cuda.is_available() else None)
    
  try:  
    results_filename = "{}_{}".format(EXPERIMENT, rerank_model.replace("/", '-'))
    results_file = output_file(experiment_name=results_filename, output_dir=output_dir)
    results_file.write("EXPERIMENT: {}\n\n".format(EXPERIMENT))
    results_file.write("INPUT PARAMETERS:\n")
    results_file.write("path_cord_data: {}\n".format(path_collection_data))
    results_file.write("path_tweet_data: {}\n".format(path_query_data))
    results_file.write("output_dir: {}\n".format(output_dir))
    results_file.write("rerank_model: {}\n".format(rerank_model))
    results_file.write("rerank_k: {}\n".format(rerank_k))
    results_file.write("mrr_k: {}\n\n".format(list_mrr_k))
    
    ## Retrieve
    df_query = retrieve(df_collection=df_collection, 
                        df_query=df_query, 
                        f=results_file,
                        device=device,
                        mrr_k = list_mrr_k) 

    ## Rerank
    df_query = rerank(df_collection=df_collection, 
                      df_query=df_query, 
                      f=results_file,
                      device=device,
                      rerank_model=rerank_model,
                      rerank_k=rerank_k,
                      mrr_k =list_mrr_k)

  finally:
    results_file.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="CLEF CheckThat! Task 4B")

  parser.add_argument("path_cord_data", help="Filepath for CORD dataset (.pkl file).")
  parser.add_argument("path_tweet_data", help="Filepath for Tweet dataset (.tsv file)")
  parser.add_argument("--output_dir", help="Output directory. Defaults to 'results.'")
  parser.add_argument("--rerank_model", help="Cross-encoder reranking model.")
  parser.add_argument("--mrr_k", help="List of MRR@K results to return. Defaults to [1, 5, 10].")
  parser.add_argument("--rerank_k", help="Number of items to pull back for re-ranking with Cross-Encoder. Defaults to 10.")

  args = parser.parse_args()
  
  output_dir = args.output_dir if args.output_dir else "results"
  rerank_k = args.rerank_k if args.rerank_k else 10
  rerank_model = args.rerank_model if args.rerank_model else "cross-encoder/ms-marco-MiniLM-L-6-v2"
  mrr_k = list(args.mrr_k) if args.mrr_k else [1, 5, 10]
  
  main(path_collection_data=args.path_cord_data,
       path_query_data=args.path_tweet_data,
       output_dir=output_dir,
       rerank_model=rerank_model,
       rerank_k=rerank_k,
       list_mrr_k=mrr_k)
