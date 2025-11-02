import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


def load_data(tokenizer, texts, max_length=512):
    """Tokenize text and convert into a TensorDataset."""
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    input_ids = torch.tensor(encodings['input_ids'])
    attention_mask = torch.tensor(encodings['attention_mask'])
    return TensorDataset(input_ids, attention_mask)


def evaluate(model, dataloader, device):
    """Run evaluation and return predictions + probabilities."""
    model.eval()
    preds, probs = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1).cpu().numpy()

            batch_preds = np.argmax(probabilities, axis=1)
            preds.extend(batch_preds)
            probs.extend(probabilities)

    return np.array(preds), np.array(probs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔹 Using device: {device}")

    # Load fine-tuned model and tokenizer
    model_path = "./bert_sentiment_model"
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.to(device)

    # Load dataset
    df = pd.read_csv("cleaned_balanced_reviews.csv")
    print("🧾 Columns in dataset:", df.columns.tolist())  # 👈 helps confirm the right column

    # Convert ratings into sentiment labels
    df["sentiment"] = df["rating"].apply(lambda r: 0 if r <= 2 else 1 if r == 3 else 2)

    # Validation split (20%)
    val_df = df.sample(frac=0.2, random_state=42)

    # ✅ Auto-detect review column (based on common names)
    possible_cols = ["cleaned_review", "review", "text", "cleaned_text"]
    review_col = next((col for col in possible_cols if col in val_df.columns), None)

    if review_col is None:
        raise KeyError(
            f"❌ No valid review text column found! Available columns: {list(val_df.columns)}"
        )

    print(f"✅ Using '{review_col}' as the review text column.")

    val_texts = val_df[review_col].astype(str).tolist()
    val_labels = val_df["sentiment"].tolist()

    val_dataset = load_data(tokenizer, val_texts)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Run evaluation
    preds, probs = evaluate(model, val_loader, device)

    # Metrics
    accuracy = accuracy_score(val_labels, preds)
    print(f"\n✅ Accuracy: {accuracy:.4f}\n")

    report = classification_report(
        val_labels, preds, target_names=["Negative", "Neutral", "Positive"]
    )
    print("📋 Classification Report:\n", report)

    # Compute AUC-ROC (for multi-class)
    try:
        auc = roc_auc_score(val_labels, probs, multi_class="ovr")
        print(f"💠 AUC-ROC: {auc:.4f}")
    except Exception as e:
        print("⚠️ AUC-ROC could not be computed:", e)

    results_df = pd.DataFrame({
        "review": val_texts,
        "true_label": val_labels,
        "pred_label": preds
    })
    results_df.to_csv("evaluation_results.csv", index=False)
    print("💾 Saved predictions to evaluation_results.csv")

    # Confusion Matrix Visualization
    cm = confusion_matrix(val_labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Neutral", "Positive"],
        yticklabels=["Negative", "Neutral", "Positive"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("📊 Confusion matrix saved as confusion_matrix.png")


if __name__ == "__main__":
    main()
