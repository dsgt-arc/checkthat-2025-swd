import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

if __name__ == "__main__":
    predictions_df = pd.read_csv("gpt-4-predictions.tsv", sep='\t')
    labels_df = pd.read_csv("ct_train_data.tsv", sep='\t')

    # Ensure proper alignment of predictions and labels by index
    merged_df = pd.merge(predictions_df, labels_df, on='index')

    merged_df = merged_df[~merged_df['prediction'].str.contains("Error")]

    # Convert predictions and labels to list format for metrics computation
    preds = merged_df['prediction'].apply(eval).tolist()
    labels = merged_df['labels'].apply(eval).tolist()

    # Compute accuracy metrics
    accuracy = accuracy_score(labels, preds)
    classification_metrics = classification_report(labels, preds, zero_division=0)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_metrics}")