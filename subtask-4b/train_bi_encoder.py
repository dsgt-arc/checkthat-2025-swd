import subprocess
import sys
import os

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cu121"
])

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
import pandas as pd
from datasets import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import EarlyStoppingCallback
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
logging.basicConfig(filename=log_file, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Parameters
model_name = "msmarco-distilbert-base-v4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load corpus
df_cord = pd.read_pickle("subtask4b_collection_data.pkl")
df_cord['full_text'] = df_cord['title'] + " " + df_cord['abstract']

# Load gold examples
df_train = pd.read_csv("subtask4b_query_tweets_train.tsv", sep='\t')
df_train = df_train.merge(df_cord[['cord_uid', 'title', 'abstract']], on='cord_uid', how='left')
df_train['full_text'] = df_train['title'] + " " + df_train['abstract']
gold_examples = [InputExample(texts=[row['tweet_text'], row['full_text']], label=1.0) for _, row in df_train.iterrows()]

# Load silver examples
with open("cache/silver_augmented.pkl", "rb") as f:
    raw_silver_data = pickle.load(f)
silver_examples = [ex for ex in raw_silver_data if ex is not None]

# Dev set for evaluation
df_dev = pd.read_csv("subtask4b_query_tweets_dev.tsv", sep='\t')
queries = {str(row['post_id']): row['tweet_text'] for _, row in df_dev.iterrows()}
corpus = {row['cord_uid']: f"{row['title']} {row['abstract']}" for _, row in df_cord.iterrows()}
relevant_docs = {str(row['post_id']): set([row['cord_uid']]) for _, row in df_dev.iterrows()}

evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="clef2025-dev-eval",
    mrr_at_k=[100]
)

### ===== PHASE 1: Train on GOLD ONLY =====

logging.info("=== Phase 1: Training on GOLD examples ===")
output_dir_gold = f"output/phase1_gold_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
train_dataset_gold = Dataset.from_dict({
    "text_0": [ex.texts[0] for ex in gold_examples],
    "text_1": [ex.texts[1] for ex in gold_examples],
    "label": [ex.label for ex in gold_examples]
})

model = SentenceTransformer(model_name, device=device)
args_gold = SentenceTransformerTrainingArguments(
    output_dir=output_dir_gold,
    num_train_epochs=6,
    per_device_train_batch_size=16,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=100,
    run_name="gold-phase",
    load_best_model_at_end=True,
    metric_for_best_model="eval_clef2025-dev-eval_cosine_mrr@100",
    greater_is_better=True
)

trainer_gold = SentenceTransformerTrainer(
    model=model,
    args=args_gold,
    train_dataset=train_dataset_gold,
    loss=losses.CosineSimilarityLoss(model=model),
    evaluator=evaluator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer_gold.train()
trainer_gold.evaluate()

# Save gold-trained model
model.save(output_dir_gold + "/final")
logging.info(f"Phase 1 complete. Model saved to: {output_dir_gold}/final")

### ===== PHASE 2: Fine-tune on SILVER ONLY =====

logging.info("=== Phase 2: Fine-tuning on SILVER examples ===")
output_dir_silver = f"output/phase2_silver_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Reload model from gold checkpoint
model = SentenceTransformer(output_dir_gold + "/final", device=device)

train_dataset_silver = Dataset.from_dict({
    "text_0": [ex.texts[0] for ex in silver_examples],
    "text_1": [ex.texts[1] for ex in silver_examples],
    "label": [ex.label for ex in silver_examples]
})

args_silver = SentenceTransformerTrainingArguments(
    output_dir=output_dir_silver,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=100,
    run_name="silver-phase",
    load_best_model_at_end=True,
    metric_for_best_model="eval_clef2025-dev-eval_cosine_mrr@100",
    greater_is_better=True
)

trainer_silver = SentenceTransformerTrainer(
    model=model,
    args=args_silver,
    train_dataset=train_dataset_silver,
    loss=losses.CosineSimilarityLoss(model=model),
    evaluator=evaluator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer_silver.train()
trainer_silver.evaluate()

# Save silver-finetuned model
model.save(output_dir_silver + "/final")
logging.info(f"Phase 2 complete. Model saved to: {output_dir_silver}/final")

# === PHASE 2: Fine-tuning with HARD NEGATIVES ===

logging.info("=== Phase 2: Fine-tuning with HARD NEGATIVES ===")
output_dir_hardneg = f"output/phase2_hardneg_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Reload best model from Phase 1
model = SentenceTransformer(output_dir_gold + "/final", device=device)

# Sample hard negatives
logging.info("Generating hard negatives...")
cord_uids = df_cord['cord_uid'].tolist()
gold_with_negatives = []

for row in df_train.itertuples():
    # Positive pair
    pos_text = f"{row.title} {row.abstract}"
    gold_with_negatives.append(InputExample(texts=[row.tweet_text, pos_text], label=1.0))

    # Negative sample: pick a random non-matching doc
    while True:
        neg_uid = random.choice(cord_uids)
        if neg_uid != row.cord_uid:
            neg_doc = df_cord[df_cord['cord_uid'] == neg_uid].iloc[0]
            neg_text = f"{neg_doc['title']} {neg_doc['abstract']}"
            gold_with_negatives.append(InputExample(texts=[row.tweet_text, neg_text], label=0.0])
            break

# Create dataset
train_dataset_hardneg = Dataset.from_dict({
    "text_0": [ex.texts[0] for ex in gold_with_negatives],
    "text_1": [ex.texts[1] for ex in gold_with_negatives],
    "label": [ex.label for ex in gold_with_negatives]
})

# Training arguments
args_hardneg = SentenceTransformerTrainingArguments(
    output_dir=output_dir_hardneg,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_ratio=0.1,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=100,
    run_name="phase2-hardneg",
    load_best_model_at_end=True,
    metric_for_best_model="eval_clef2025-dev-eval_cosine_mrr@100",
    greater_is_better=True
)

# Trainer
trainer_hardneg = SentenceTransformerTrainer(
    model=model,
    args=args_hardneg,
    train_dataset=train_dataset_hardneg,
    loss=losses.CosineSimilarityLoss(model=model),
    evaluator=evaluator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train + evaluate
trainer_hardneg.train()
trainer_hardneg.evaluate()

# Save final model
model.save(output_dir_hardneg + "/final")
logging.info(f"Phase 2 (hard negatives) complete. Model saved to: {output_dir_hardneg}/final")