import argparse
import numpy as np
import pandas as pd
import os

from bm25 import BM25
from util import get_performance_mrr, output_file

EXPERIMENT = "00_baseline"


def baseline(df_collection, df_query, f, mrr_k = [1, 5, 10]):
  # Create the BM25 corpus
  corpus = df_collection[:][['title', 'abstract']].apply(lambda x: f"{x['title']} {x['abstract']}", axis=1).tolist()
  cord_uids = df_collection[:]['cord_uid'].tolist()

  bm25 = BM25(corpus=corpus, cord_uids=cord_uids)

  # Retrieve topk candidates using the BM25 model
  df_query['bm25_topk'] = df_query['tweet_text'].apply(lambda x: bm25.get_top_cord_uids(x))

  results = get_performance_mrr(df_query, 
                                col_gold='cord_uid', 
                                col_pred='bm25_topk', 
                                list_k = mrr_k,
                                title='Baseline - BM25')
  
  f.write("EXPERIMENT {} BASELINE RESULTS:\n".format(EXPERIMENT))
  f.write(str(results))
  
  return df_query


def main(path_collection_data, path_query_data, output_dir, list_mrr_k):
  
  df_collection = pd.read_pickle(path_collection_data)
  df_query = pd.read_csv(path_query_data, sep = '\t')
   
  try:  
    results_file = output_file(experiment_name=EXPERIMENT, output_dir=output_dir)
    results_file.write("EXPERIMENT: {}\n\n".format(EXPERIMENT))
    results_file.write("INPUT PARAMETERS:\n")
    results_file.write("path_cord_data: {}\n".format(path_collection_data))
    results_file.write("path_tweet_data: {}\n".format(path_query_data))
    results_file.write("output_dir: {}\n".format(output_dir))
    results_file.write("mrr_k: {}\n\n".format(list_mrr_k))
    
    df_query = baseline(df_collection=df_collection, 
            df_query=df_query, 
            f=results_file,
            mrr_k = list_mrr_k)
    
  finally:
    results_file.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="CLEF CheckThat! Task 4B")

  parser.add_argument("path_cord_data", help="Filepath for CORD dataset (.pkl file).")
  parser.add_argument("path_tweet_data", help="Filepath for Tweet dataset (.tsv file)")
  parser.add_argument("--output_dir", help="Output directory. Defaults to results")
  parser.add_argument("--mrr_k", help="List of MRR@K results to return. Defaults to [1, 5, 10].")

  args = parser.parse_args()
  
  output_dir = args.output_dir if args.output_dir else "results"
  mrr_k = list(args.mrr_k) if args.mrr_k else [1, 5, 10]
  
  main(path_collection_data=args.path_cord_data,
       path_query_data=args.path_tweet_data,
       output_dir=output_dir,
       list_mrr_k=mrr_k)
