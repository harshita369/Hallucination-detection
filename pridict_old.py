"""
=============================================================
  Hallucination Detector — Inference Script
  Loads the trained RoBERTa model and classifies sentences
  as Factual or Hallucinated.

  Usage:
    python predict.py "The Eiffel Tower is in Berlin."
    python predict.py   (interactive mode)
=============================================================
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── CONFIG ─────────────────────────────────────────────────
MODEL_DIR  = "./detector_model/best"   # change to "./detector_model/final" if needed
MAX_LENGTH = 128

# ── LOAD MODEL ─────────────────────────────────────────────
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2)
model.to(device)
model.eval()

print(f"Model loaded from: {MODEL_DIR}")
print(f"Device: {device}\n")


def predict(statement, evidence=""):
    """Classify a statement as Factual or Hallucinated."""
    encoding = tokenizer(
        str(statement),
        str(evidence),
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    factual_prob = probs[0].item()
    hallucination_prob = probs[1].item()
    label = "Hallucinated" if hallucination_prob > 0.5 else "Factual"

    return {
        "label": label,
        "confidence": round(max(factual_prob, hallucination_prob) * 100, 2),
        "factual_prob": round(factual_prob * 100, 2),
        "hallucination_prob": round(hallucination_prob * 100, 2),
    }


def display_result(statement, result):
    flag = "!!" if result["label"] == "Hallucinated" else "OK"
    print(f"  [{flag}] {result['label']} (confidence: {result['confidence']}%)")
    print(f"       Factual: {result['factual_prob']}%  |  Hallucinated: {result['hallucination_prob']}%")


# ── MAIN ───────────────────────────────────────────────────
if __name__ == "__main__":

    # If a sentence was passed as a command-line argument
    if len(sys.argv) > 1:
        sentence = " ".join(sys.argv[1:])
        print(f"Input: {sentence}\n")
        result = predict(sentence)
        display_result(sentence, result)
        sys.exit(0)

    # Interactive mode
    print("=" * 55)
    print("  Hallucination Detector — Interactive Mode")
    print("  Type a sentence and press Enter.")
    print("  Type 'quit' to exit.")
    print("=" * 55)

    while True:
        try:
            sentence = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not sentence or sentence.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        result = predict(sentence)
        display_result(sentence, result)