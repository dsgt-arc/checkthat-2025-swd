import os

# Evaluate retrieved candidates using MRR@k
def get_performance_mrr(data, col_gold, col_pred, list_k = [1, 5, 10]):
  d_performance = {}
  for k in list_k:
      data["in_topx"] = data.apply(lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) if x[col_gold] in [i for i in x[col_pred][:k]] else 0), axis=1)
      d_performance[k] = data["in_topx"].mean()
        
  return d_performance

def retrieve_paper(df_collection, paper_ids):
  paper_dict = {}
  for id in paper_ids:
    paper_data = df_collection[df_collection['cord_uid'] == id]
    title = paper_data.iloc[0]['title']
    abstract = paper_data.iloc[0]['abstract']
    paper_dict[id] = {'title': title, 'abstract': abstract}
    
  return paper_dict

def output_file(experiment_name, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    f = open("{}/subtask4b_{}.txt".format(output_dir, experiment_name), "w")
    
    return f