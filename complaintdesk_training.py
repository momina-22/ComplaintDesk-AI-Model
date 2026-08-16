import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# ============================================================
# 1. Dataset Class
# ============================================================

class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================
# 2. Prediction Helper
# ============================================================

def test_prediction(text, model, tokenizer, le, device):
    """Quick prediction test on single sample."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()

    priority = le.inverse_transform([pred_id])[0]
    confidence = probs[0][pred_id].item() * 100

    scores = {
        le.inverse_transform([i])[0]: f"{probs[0][i].item() * 100:.1f}%"
        for i in range(len(le.classes_))
    }

    print(f"Text      : {text}")
    print(f"Prediction: {priority}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"All scores: {scores}")
    print("-" * 60)


# ============================================================
# 3. Main Training Execution
# ============================================================

def main():
    dataset_path = Path("my_dataset.csv")
    output_dir = Path("complaintNLP_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file '{dataset_path}' not found.")

    # Load and clean data
    df = pd.read_csv(dataset_path)
    df = df[["complaint_text", "priority"]].dropna()

    # Show dataset distribution
    print("Label distribution in your dataset:")
    print(df["priority"].value_counts())
    print("\nAs percentages:")
    print(df["priority"].value_counts(normalize=True) * 100)
    print("\n" + "=" * 60)

    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["priority"])

    # Initialize Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        df["complaint_text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42,
    )

    train_dataset = ComplaintDataset(X_train, y_train, tokenizer)
    val_dataset = ComplaintDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Load DistilBERT model
    num_labels = len(le.classes_)
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on: {device}")

    # Training setup
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 3

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total * 100
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Accuracy: {accuracy:.2f}%")

    # Save model and tokenizer
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mapping
    label_map = dict(zip(le.transform(le.classes_).tolist(), le.classes_.tolist()))
    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print("Model and label_map.json saved successfully!")
    print("\n" + "=" * 60)
    print("Running quick prediction tests:")
    print("=" * 60)

    model.eval()
    test_prediction("My internet has been down for 3 days and I work from home", model, tokenizer, le, device)
    test_prediction("I want to update my billing address", model, tokenizer, le, device)
    test_prediction("The service is okay but I have a small question about my plan", model, tokenizer, le, device)
    test_prediction("There is the gas leakage in block A ", model, tokenizer, le, device)
    test_prediction("Can I get a copy of last month's invoice please", model, tokenizer, le, device)


if __name__ == "__main__":
    main()