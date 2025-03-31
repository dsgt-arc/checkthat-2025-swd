import torch
from models.encoder_classifier import Classifier
import polars as pl
import ast
from sklearn.metrics import f1_score, classification_report

def macro_f1_fn(preds: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(preds) > 0.5).int()
    return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")

def main():
    # 1. Define your model path directory (no '.pth' at the end if it's a folder)
    model_path = "C:/Users/leoch/Downloads/CLEF/4a/fine_tuned_models/microsoft-deberta-v3-base-classifier.pth"

    # 2. Load the fine-tuned model
    model = Classifier.load(model_path)

    # 3. Read the test data (CHANGE THIS to whichever file you want to evaluate)
    test_data_path = "C:/Users/leoch/Downloads/CLEF/4a/ct_train_data.tsv"
    df_test = pl.read_csv(test_data_path, separator="\t")

    # Ensure your 'label' column is named consistently
    # If your file has columns: [sentence_id, sentence, label],
    # then we do:
    if "labels" not in df_test.columns:
        raise ValueError("No 'label' column found in the test dataset!")

    # 4. Map textual labels (e.g. 'OBJ'/'SUBJ') to integers (0 or 1)
    #    Here, we assume 'OBJ' -> 0 and 'SUBJ' -> 1
    ''''
    label_map = {"OBJ": 0, "SUBJ": 1}
    test_labels = [label_map[label] for label in df_test["labels"]]'
    '''
    test_labels = [ast.literal_eval(label) for label in df_test["labels"]]

    # 5. Tokenize all sentences
    test_tokens = model.tokenizer(
        df_test["text"].to_list(),
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True
    )

    # 6. Inference: get logits for the entire dataset
    # 6. Run model inference + sigmoid for multilabel outputs
    with torch.no_grad():
        logits = model(
            test_tokens["input_ids"], 
            test_tokens["attention_mask"]
        )
        probs = torch.sigmoid(logits)
        preds_tensor = (probs > 0.5).int()

    '''                
    # Predicted class indices
    preds_tensor = torch.argmax(logits, dim=1)

    # 7. Convert predictions to Python list
    preds = preds_tensor.tolist()  # list of 0/1
    ''' 

    # 7. Convert labels to tensor
    test_labels_tensor = torch.tensor(test_labels).int()

    '''
    # 8. Compute Accuracy & Classification Report
    accuracy = accuracy_score(test_labels, preds)
    print(f"Accuracy: {accuracy:.3f}")

    # The 'target_names' must match label_map's order
    # If 0 = OBJ, 1 = SUBJ:
    report = classification_report(
        test_labels, 
        preds, 
        target_names=["OBJ", "SUBJ"]
    )
    print(report)

    # 9. Optionally print first 10 predictions in text form
    label_map_inv = {0: "OBJ", 1: "SUBJ"}
    print("First 10 predictions:")
    print([label_map_inv[p] for p in preds[:10]])'
    '''

    # 8. Compute macro F1 score for multi-label classification 
    #f1 = f1_score(test_labels_tensor, preds_tensor, average="macro")
    f1 = macro_f1_fn(logits, test_labels_tensor)
    print(f"Macro F1 Score: {f1:.3f}")

    # 9. Optional: Print classification report
    target_names = ["cat_1", "cat_2", "cat_3"]
    print("\nClassification Report:")
    print(classification_report(test_labels_tensor, preds_tensor, target_names=target_names, zero_division=0))

    # 10. Print first 10 predictions and actuals 
    print("\nFirst 10 multi-label predictions:")
    for i in range(min(10, len(df_test))):
        print(f"Text:      {df_test['text'][i]}")
        pred_labels = [target_names[j] for j, val in enumerate(preds_tensor[i]) if val == 1]
        true_labels = [target_names[j] for j, val in enumerate(test_labels_tensor[i]) if val == 1]
        print(f"Predicted: {pred_labels}")
        print(f"Actual:    {true_labels}")
        print("-" * 50)

if __name__ == "__main__":
    main()
