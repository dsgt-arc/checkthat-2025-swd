import subprocess
import sys
import os

# Install required packages
required_packages = [
    "tqdm",
    "torch",
    "nlpaug",
    "pandas",
    "datasets",
    "sentence-transformers",
    "accelerate>=0.26.0"
]
for package in required_packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Imports
import logging
from datetime import datetime
import tqdm
import torch
import pickle
import nlpaug.augmenter.word as naw
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Parameters
model_name = "bert-base-uncased"
silver_cache_path = "cache/silver_augmented.pkl"
os.makedirs("cache", exist_ok=True)

# Load document corpus
df_cord = pd.read_pickle("subtask4b_collection_data.pkl")
df_cord['doc_id'] = df_cord['cord_uid']
df_cord['full_text'] = df_cord['title'] + " " + df_cord['abstract']

# Merge training queries with document texts
df_train = pd.read_csv("subtask4b_query_tweets_train.tsv", sep='\t')
df_train = df_train.merge(df_cord[['cord_uid', 'title', 'abstract']], on='cord_uid', how='left')
df_train['full_text'] = df_train['title'] + " " + df_train['abstract']

# Load or initialize silver_data
if os.path.exists(silver_cache_path):
    print("Load cached silver_data")
    with open(silver_cache_path, "rb") as f:
        silver_data = pickle.load(f)
else:
    silver_data = []

# Augment using ContextualWordEmbsAug with resume support
aug = naw.ContextualWordEmbsAug(model_path=model_name, action="insert")
start_idx = len(silver_data)

for i, row in tqdm.tqdm(df_train.iterrows(), total=len(df_train), desc="Augment training data"):
    if i < start_idx:
        continue
    try:
        aug_q, aug_d = aug.augment([row['tweet_text'], row['full_text']])
        silver_data.append((aug_q, aug_d))  # Save raw tuple
    except Exception as e:
        logging.warning(f"\nAugmentation failed at index {i}: {e}")
        silver_data.append(None)

    if i % 100 == 0 or i == len(df_train) - 1:
        with open(silver_cache_path, "wb") as f:
            pickle.dump(silver_data, f)
        print(f"\nSaved silver_data at index {i}")

print("Finished augmenting and saving silver_data.")