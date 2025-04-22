import torch
from models.encoder_classifier import Classifier
import polars as pl
import pandas as pd
import random
import numpy as np
import ast
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

def macro_f1_fn(preds: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(preds) > 0.5).int()
    return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")

def main():

    def set_seed(seed=88):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(88)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define your model path directory (no '.pth' at the end if it's a folder)
    model_path = "C:/Users/leoch/Downloads/CLEF/4a/fine_tuned_models/microsoft-deberta-v3-base-classifier.pth"

    # Load the fine-tuned model
    model = Classifier.load(model_path)
    model.eval()

    # Read the test data (CHANGE THIS to whichever file you want to evaluate)
    test_data_path = "C:/Users/leoch/Downloads/CLEF/4a/ct_dev.tsv"
    df_test = pl.read_csv(test_data_path, separator="\t")

    # Ensure your 'label' column is named consistently
    # If your file has columns: [sentence_id, sentence, label],
    # then we do:
    if "labels" not in df_test.columns:
        raise ValueError("No 'label' column found in the test dataset!")

    # 4. Map textual labels 
    test_labels = [ast.literal_eval(label) for label in df_test["labels"]]

    # 5. Tokenize all sentences
    texts = df_test["text"].to_list()
    lengths = [len(model.tokenizer(text, truncation=False)["input_ids"]) for text in tqdm(texts)]
    max_len = int(np.percentile(lengths, 95))

    test_tokens = model.tokenizer(
        df_test["text"].to_list(),
        padding=True,
        truncation=True, 
        max_length=max_len,
        return_tensors="pt",
        return_attention_mask=True
    )

    input_ids = test_tokens["input_ids"].to(device)
    attention_mask = test_tokens["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        preds_tensor = (probs > 0.5).int()

    # Convert labels to tensor
    test_labels_tensor = torch.tensor(test_labels).int()

    # Compute macro F1 score for multi-label classification 
    f1 = macro_f1_fn(logits, test_labels_tensor)
    print(f"Macro F1 Score: {f1:.3f}")

    # Print classification report
    target_names = ["cat_1", "cat_2", "cat_3"]
    print("\nClassification Report:")
    print(classification_report(
        test_labels_tensor.cpu(), 
        preds_tensor.cpu(), 
        target_names=target_names, 
        zero_division=0
    ))

    # Print first 10 predictions and actuals 
    print("\nFirst 10 multi-label predictions:")
    for i in range(min(10, len(df_test))):
        print(f"Text:      {df_test['text'][i]}")
        pred_labels = [target_names[j] for j, val in enumerate(preds_tensor[i]) if val == 1]
        true_labels = [target_names[j] for j, val in enumerate(test_labels_tensor[i]) if val == 1]
        print(f"Predicted: {pred_labels}")
        print(f"Actual:    {true_labels}")
        print("-" * 50)

    # Save predictions in required format
    preds_df = pd.DataFrame({
        "index": df_test["index"].to_list() if "index" in df_test.columns else list(range(len(df_test))),
        "cat1_pred": preds_tensor[:, 0].tolist(),
        "cat2_pred": preds_tensor[:, 1].tolist(),
        "cat3_pred": preds_tensor[:, 2].tolist()
    })

    preds_df.to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()
