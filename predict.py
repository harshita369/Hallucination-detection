"""
=============================================================
  Hallucination Detector — Improved Inference Script
  Loads ALL trained models and runs full pipeline.

  IMPROVEMENTS over v1:
  1. Loads tuned threshold from Phase 3
  2. Runs NLI verification alongside detection
  3. Fetches Wikipedia evidence automatically
  4. Runs correction engine if hallucinated
  5. Shows weighted combined score

  Usage:
    python predict.py "The Eiffel Tower is in Berlin."
    python predict.py   (interactive mode)
=============================================================
"""

import sys
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration
)

# ── CONFIG ─────────────────────────────────────────────────
DETECTOR_PATH  = "./detector_model/best"
NLI_PATH       = "./nli_model/best"
CORRECTOR_PATH = "./corrector_model/best"
MAX_LENGTH     = 128

# ── DEVICE ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── LOAD THRESHOLD ─────────────────────────────────────────
threshold = 0.5
t_file = os.path.join(DETECTOR_PATH, "threshold.txt")
if os.path.exists(t_file):
    with open(t_file) as f:
        threshold = float(f.read().strip())

# ── LOAD MODELS ────────────────────────────────────────────
print("Loading models...")

det_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_PATH)
det_model     = AutoModelForSequenceClassification.from_pretrained(DETECTOR_PATH).to(device)
det_model.eval()

nli_tokenizer = AutoTokenizer.from_pretrained(NLI_PATH)
nli_model     = AutoModelForSequenceClassification.from_pretrained(NLI_PATH).to(device)
nli_model.eval()

cor_tokenizer = T5Tokenizer.from_pretrained(CORRECTOR_PATH, legacy=False)
cor_model     = T5ForConditionalGeneration.from_pretrained(CORRECTOR_PATH).to(device)
cor_model.eval()

# ── LOAD RETRIEVER ─────────────────────────────────────────
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from retriever import get_evidence
    RETRIEVER = True
except:
    RETRIEVER = False
    def get_evidence(claim, verbose=False):
        return {"evidence": "", "found": False}

print(f"✅ Models loaded | Device: {device} | Threshold: {threshold}")
print(f"   Wikipedia retrieval: {'enabled' if RETRIEVER else 'disabled'}\n")


# ── INFERENCE FUNCTIONS ────────────────────────────────────

def detect(statement, evidence=""):
    enc = det_tokenizer(str(statement), str(evidence),
                        max_length=MAX_LENGTH, truncation=True,
                        padding="max_length", return_tensors="pt")
    with torch.no_grad():
        out = det_model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
    probs = torch.softmax(out.logits, dim=1)[0]
    h_prob = probs[1].item()
    return {
        "label"     : "Hallucinated" if h_prob > threshold else "Factual",
        "h_prob"    : round(h_prob * 100, 2),
        "f_prob"    : round(probs[0].item() * 100, 2),
        "raw"       : h_prob
    }


def nli_check(premise, hypothesis):
    enc = nli_tokenizer(str(premise), str(hypothesis),
                        max_length=MAX_LENGTH, truncation=True,
                        padding="max_length", return_tensors="pt")
    with torch.no_grad():
        out = nli_model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
    probs     = torch.softmax(out.logits, dim=1)[0]
    label_idx = torch.argmax(probs).item()
    label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
    return {
        "label"         : label_map[label_idx],
        "entailment"    : round(probs[0].item() * 100, 2),
        "neutral"       : round(probs[1].item() * 100, 2),
        "contradiction" : round(probs[2].item() * 100, 2),
        "raw_contra"    : probs[2].item()
    }


def correct(statement, evidence=""):
    if evidence:
        prompt = (f"Given the following evidence: '{evidence}' "
                  f"The statement '{statement}' is factually incorrect. "
                  f"Write a complete corrected factual sentence:")
    else:
        prompt = (f"The following statement is factually incorrect: '{statement}' "
                  f"Write a complete corrected factual sentence:")

    enc = cor_tokenizer(prompt, max_length=256, truncation=True,
                        padding="max_length", return_tensors="pt")
    with torch.no_grad():
        ids = cor_model.generate(
            input_ids      = enc["input_ids"].to(device),
            attention_mask = enc["attention_mask"].to(device),
            max_length     = 128, min_length=8,
            num_beams      = 4, length_penalty=1.5,
            early_stopping = True, no_repeat_ngram_size=3
        )
    return cor_tokenizer.decode(ids[0], skip_special_tokens=True)


def full_pipeline(statement):
    """Run the full 4-step pipeline on a statement."""

    # Step 1 — Fetch evidence
    evidence = ""
    source   = "none"
    if RETRIEVER:
        ev = get_evidence(statement, verbose=False)
        if ev.get("found"):
            evidence = ev["evidence"]
            source   = ev.get("source", "Wikipedia")

    # Step 2 — Detection
    det = detect(statement, evidence)

    # Step 3 — NLI verification
    nli = None
    if evidence:
        nli = nli_check(evidence, statement)

    # Step 4 — Weighted combined verdict
    c_prob = nli["raw_contra"] if nli else det["raw"]
    weighted = 0.6 * det["raw"] + 0.4 * c_prob

    if weighted > 0.6:
        verdict    = "Hallucinated"
        confidence = "HIGH"
    elif weighted > 0.35:
        verdict    = "Hallucinated"
        confidence = "MEDIUM"
    else:
        verdict    = "Factual"
        confidence = "LOW"

    # Step 5 — Correction if needed
    corrected = None
    if verdict == "Hallucinated":
        corrected = correct(statement, evidence)

    return {
        "statement"  : statement,
        "evidence"   : evidence,
        "source"     : source,
        "detection"  : det,
        "nli"        : nli,
        "verdict"    : verdict,
        "confidence" : confidence,
        "weighted"   : round(weighted * 100, 2),
        "corrected"  : corrected,
    }


def display(result):
    """Pretty print pipeline result."""
    flag = "⚠  HALLUCINATED" if result["verdict"] == "Hallucinated" else "✅ FACTUAL"
    print(f"\n  {'─'*55}")
    print(f"  {flag}")
    print(f"  {'─'*55}")
    print(f"  Statement  : {result['statement']}")
    print(f"  Detection  : {result['detection']['h_prob']}% hallucination "
          f"(threshold: {threshold})")

    if result["nli"]:
        nli = result["nli"]
        print(f"  NLI        : {nli['label']} "
              f"(Entailment: {nli['entailment']}% | "
              f"Contradiction: {nli['contradiction']}%)")

    print(f"  Weighted   : {result['weighted']}% | Confidence: {result['confidence']}")

    if result["evidence"]:
        ev_preview = result["evidence"][:120].replace("\n", " ")
        print(f"  Evidence   : [{result['source']}] {ev_preview}...")

    if result["corrected"]:
        print(f"\n  ✏ Corrected: {result['corrected']}")

    print()


# ── MAIN ───────────────────────────────────────────────────
if __name__ == "__main__":

    # Command-line mode
    if len(sys.argv) > 1:
        statement = " ".join(sys.argv[1:])
        print(f"\nAnalyzing: {statement}")
        result = full_pipeline(statement)
        display(result)
        sys.exit(0)

    # Interactive mode
    print("=" * 57)
    print("  Hallucination Detector — Full Pipeline")
    print("  Type a statement and press Enter.")
    print("  Commands: 'quit' to exit")
    print("=" * 57)

    while True:
        try:
            statement = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not statement or statement.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        result = full_pipeline(statement)
        display(result)
