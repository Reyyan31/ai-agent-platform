"""
intent_classifier.py
--------------------
Loads the trained intent classifier once at import time and exposes
classify_intent(text) -> {"label": str, "confidence": float}.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# ── Paths (relative to this file's location) ──────────────────────────────────
_AGENT_DIR  = os.path.dirname(os.path.abspath(__file__))
_BACKEND    = os.path.abspath(os.path.join(_AGENT_DIR, "..", ".."))
_LABELS_PATH  = os.path.join(_BACKEND, "models", "labels.json")
_WEIGHTS_PATH = os.path.join(_BACKEND, "models", "intent_classifier.pt")

# ── Model architecture (must match train_classifier.py) ───────────────────────
class _IntentClassifier(nn.Module):
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


# ── Load once at import time ───────────────────────────────────────────────────
with open(_LABELS_PATH, "r") as _f:
    _LABELS: list[str] = json.load(_f)

_NUM_LABELS = len(_LABELS)

_model = _IntentClassifier(input_dim=384, num_classes=_NUM_LABELS)
_model.load_state_dict(torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True))
_model.eval()

_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Public API ─────────────────────────────────────────────────────────────────

def classify_intent(text: str) -> dict:
    """
    Embed *text* with the sentence transformer and run the classifier.

    Returns:
        {"label": <predicted label string>, "confidence": <float 0-1>}
    """
    embedding = _embedder.encode([text])                         # (1, 384)
    tensor    = torch.tensor(embedding, dtype=torch.float32)

    with torch.no_grad():
        logits      = _model(tensor)                             # (1, num_labels)
        probs       = F.softmax(logits, dim=1)                   # (1, num_labels)
        confidence, idx = torch.max(probs, dim=1)

    return {
        "label":      _LABELS[idx.item()],
        "confidence": round(confidence.item(), 4),
    }


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    examples = [
        "write a python function to binary search a sorted list",
        "what does my resume say about cloud experience",
        "create a word document summarising my skills",
        "what is 128 divided by 4",
        "echo this back: deployment complete",
    ]

    print("Intent classifier self-test\n" + "-" * 40)
    for text in examples:
        result = classify_intent(text)
        print(f"  Input : {text!r}")
        print(f"  Result: {result}")
        print()
