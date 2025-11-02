import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np
import os

MODEL_NAME = "bert-base-uncased"
MODEL_SAVE_PATH = "./bert_sentiment_model"
EPOCHS = 6
BATCH_SIZE = 8
MAX_LEN = 128
LR = 1e-5



# 1. Custom Dataset Class
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


#  2. Load and preprocess data (with neutral upsampling)
def load_data():
    df = pd.read_csv("cleaned_balanced_reviews.csv")

    # Ensure sentiment column exists
    if 'sentiment' not in df.columns:
        df["sentiment"] = df["rating"].apply(lambda r: 0 if r <= 2 else 1 if r == 3 else 2)

    print(f"✅ Original dataset: {df.shape[0]} samples")
    print(df['sentiment'].value_counts())

    neutral_df = df[df['sentiment'] == 1]
    df_balanced = pd.concat([df, neutral_df])  # duplicate neutral twice
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n✅ After upsampling neutral class:")
    print(df_balanced['sentiment'].value_counts())

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df_balanced["cleaned_review"].tolist(),
        df_balanced["sentiment"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df_balanced["sentiment"]
    )
    return train_texts, val_texts, train_labels, val_labels, df_balanced


#  3. Create data loaders
def create_data_loaders(tokenizer, train_texts, val_texts, train_labels, val_labels):
    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    return train_loader, val_loader


#  4. Evaluation helper
def evaluate(model, data_loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            preds.extend(predictions.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    acc = accuracy_score(labels, preds)
    return acc, classification_report(labels, preds, target_names=['Negative', 'Neutral', 'Positive'])


#  5. Training loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)

    train_texts, val_texts, train_labels, val_labels, df_balanced = load_data()
    train_loader, val_loader = create_data_loaders(tokenizer, train_texts, val_texts, train_labels, val_labels)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # ✅ Compute class weights
    class_counts = df_balanced['sentiment'].value_counts().sort_index()
    total = sum(class_counts)
    class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float)
    class_weights = class_weights / class_weights.sum()

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    print("🧭 Using class weights:", class_weights)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        print(f"\n🧠 Epoch {epoch + 1}/{EPOCHS}")
        loop = tqdm(train_loader, desc="Training", leave=False)

        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"📉 Average training loss: {avg_train_loss:.4f}")

        val_acc, val_report = evaluate(model, val_loader, device)
        print(f"✅ Validation Accuracy: {val_acc:.4f}")
        print(val_report)

        # Save after each epoch
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        print(f"💾 Model saved at {MODEL_SAVE_PATH}")

    print("🎯 Training complete!")


if __name__ == "__main__":
    train()
