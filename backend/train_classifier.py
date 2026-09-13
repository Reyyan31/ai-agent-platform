import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    # ── Paths ──────────────────────────────────────────────────────────────
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH  = os.path.join(BASE_DIR, "data", "intent_training_data.csv")
    MODEL_DIR  = os.path.join(BASE_DIR, "models")
    LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
    WEIGHTS_PATH = os.path.join(MODEL_DIR, "intent_classifier.pt")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples with labels: {sorted(df['label'].unique())}")

    # ── Encode text → embeddings (384-dim) ────────────────────────────────
    print("Encoding sentences with SentenceTransformer...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = st_model.encode(df["text"].tolist(), show_progress_bar=True)

    # ── Encode labels → integers ───────────────────────────────────────────
    le = LabelEncoder()
    int_labels = le.fit_transform(df["label"].tolist())
    label_names = sorted(le.classes_.tolist())          # sorted list of unique labels
    num_labels  = len(label_names)

    # Save label list so the inference side can decode predictions
    with open(LABELS_PATH, "w") as f:
        json.dump(label_names, f)
    print(f"Saved {num_labels} labels to {LABELS_PATH}: {label_names}")

    # ── Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, int_labels,
        test_size=0.2,
        random_state=42,
        stratify=int_labels,
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── PyTorch model ──────────────────────────────────────────────────────
    class IntentClassifier(nn.Module):
        def __init__(self, input_dim: int, num_classes: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    model = IntentClassifier(input_dim=384, num_classes=num_labels)

    # ── Convert to tensors ─────────────────────────────────────────────────
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    # ── Training ───────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\nTraining for 100 epochs...")
    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss   = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    # ── Evaluation ─────────────────────────────────────────────────────────
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits     = model(X_test_t)
        predictions = logits.argmax(dim=1).numpy()

    acc = accuracy_score(y_test, predictions)
    print(f"\nTest Accuracy: {acc * 100:.1f}%")
    print(classification_report(y_test, predictions, target_names=label_names))

    # ── Save weights ───────────────────────────────────────────────────────
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Model weights saved to {WEIGHTS_PATH}")
