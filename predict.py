"""
=============================================================
  Hallucination Detector — Full Pipeline Inference
  GPU    : RTX 3050 Mobile 4GB — Linux

  FIXES vs old pridict.py:
  1. Detection runs STATEMENT-ONLY (no evidence needed)
     — old model needed evidence and failed without it
  2. Wikipedia evidence fetched and used for NLI+correction
     only (not for the detection decision itself)
  3. Weighted combination: 60% detection + 40% NLI
  4. Threshold loaded from threshold.txt (tuned value)
  5. Correction generates full sentences (min_length=15)

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

# ── Paths ─────────────────────────────────────────────────
DETECTOR_PATH  = "./detector_model/best"
NLI_PATH       = "./nli_model/best"
CORRECTOR_PATH = "./corrector_model/best"
MAX_LEN        = 128

# ── Device ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load threshold ────────────────────────────────────────
threshold = 0.5
t_file = os.path.join(DETECTOR_PATH, "threshold.txt")
if os.path.exists(t_file):
    with open(t_file) as f:
        threshold = float(f.read().strip())

# ── Load models ───────────────────────────────────────────
print("Loading models...")

det_tok   = AutoTokenizer.from_pretrained(DETECTOR_PATH)
det_model = AutoModelForSequenceClassification.from_pretrained(DETECTOR_PATH).to(device)
det_model.eval()

nli_tok   = AutoTokenizer.from_pretrained(NLI_PATH)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_PATH).to(device)
nli_model.eval()

cor_tok   = T5Tokenizer.from_pretrained(CORRECTOR_PATH, legacy=False)
cor_model = T5ForConditionalGeneration.from_pretrained(CORRECTOR_PATH).to(device)
cor_model.eval()

# ── Load retriever ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from retriever import get_evidence
    RETRIEVER = True
except:
    RETRIEVER = False
    def get_evidence(claim, verbose=False):
        return {"evidence": "", "found": False}

print(f"✅ Models loaded | Device: {device} | Threshold: {threshold}")
print(f"   Wikipedia: {'enabled' if RETRIEVER else 'disabled'}\n")


# ── INFERENCE FUNCTIONS ───────────────────────────────────

def detect(statement):
    """
    STATEMENT-ONLY detection.
    The model was trained without evidence so it works
    purely on language patterns — no Wikipedia needed.
    """
    enc = det_tok(str(statement), max_length=MAX_LEN,
                  truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        out = det_model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
    probs  = torch.softmax(out.logits, dim=1)[0]
    h_prob = probs[1].item()
    return {
        "label"  : "Hallucinated" if h_prob > threshold else "Factual",
        "h_prob" : round(h_prob * 100, 2),
        "f_prob" : round(probs[0].item() * 100, 2),
        "raw"    : h_prob
    }


def nli_check(premise, hypothesis):
    """NLI verification — only called when evidence is available."""
    enc = nli_tok(str(premise), str(hypothesis), max_length=MAX_LEN,
                  truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        out = nli_model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
    probs = torch.softmax(out.logits, dim=1)[0]
    idx   = torch.argmax(probs).item()
    lbls  = {0:"Entailment", 1:"Neutral", 2:"Contradiction"}
    return {
        "label"        : lbls[idx],
        "entailment"   : round(probs[0].item()*100, 2),
        "neutral"      : round(probs[1].item()*100, 2),
        "contradiction": round(probs[2].item()*100, 2),
        "raw_contra"   : probs[2].item()
    }


def correct(statement, evidence=""):
    """Generate a corrected full sentence."""
    if evidence and evidence.strip():
        prompt = (f"Task: Correct the factual error in the statement below. "
                  f"Use the evidence provided. "
                  f"Write one complete corrected sentence.\n"
                  f"Evidence: {evidence.strip()}\n"
                  f"Incorrect statement: {statement}\n"
                  f"Corrected statement:")
    else:
        prompt = (f"Task: Correct the factual error in the statement below. "
                  f"Write one complete corrected sentence.\n"
                  f"Incorrect statement: {statement}\n"
                  f"Corrected statement:")

    enc = cor_tok(prompt, max_length=256, truncation=True,
                  padding="max_length", return_tensors="pt")
    with torch.no_grad():
        ids = cor_model.generate(
            input_ids            = enc["input_ids"].to(device),
            attention_mask       = enc["attention_mask"].to(device),
            max_length           = 128,
            min_length           = 15,
            num_beams            = 4,
            length_penalty       = 2.0,
            early_stopping       = True,
            no_repeat_ngram_size = 3,
            do_sample            = False
        )
    return cor_tok.decode(ids[0], skip_special_tokens=True)


def full_pipeline(statement):
    """
    Full 4-step pipeline:
    1. Detection (statement-only — always works)
    2. Wikipedia evidence fetch (optional, enhances NLI)
    3. NLI verification (only if evidence found)
    4. Correction (if hallucinated)
    """
    # Step 1 — Detection (statement only, no evidence)
    det = detect(statement)

    # Step 2 — Fetch Wikipedia evidence (for NLI + correction)
    evidence = ""
    ev_source = ""
    if RETRIEVER:
        ev = get_evidence(statement, verbose=False)
        if ev.get("found"):
            evidence  = ev.get("evidence", "")
            ev_source = ev.get("source", "Wikipedia")

    # Step 3 — NLI (only if evidence found)
    nli = None
    if evidence and evidence.strip():
        nli = nli_check(evidence, statement)

    # Step 4 — Weighted combined verdict
    # Detection is always available; NLI supplements it when evidence exists
    c_prob   = nli["raw_contra"] if nli else det["raw"]
    weighted = 0.6 * det["raw"] + 0.4 * c_prob

    if weighted > 0.6:
        verdict, conf = "Hallucinated", "HIGH"
    elif weighted > 0.35:
        verdict, conf = "Hallucinated", "MEDIUM"
    else:
        verdict, conf = "Factual", "LOW"

    # Step 5 — Correction if hallucinated
    corrected = None
    if verdict == "Hallucinated":
        corrected = correct(statement, evidence)

    return {
        "statement"  : statement,
        "evidence"   : evidence,
        "ev_source"  : ev_source,
        "detection"  : det,
        "nli"        : nli,
        "verdict"    : verdict,
        "confidence" : conf,
        "weighted"   : round(weighted * 100, 2),
        "corrected"  : corrected,
    }


def display(result):
    """Pretty-print pipeline result."""
    flag = "⚠  HALLUCINATED" if result["verdict"] == "Hallucinated" else "✅ FACTUAL"
    print(f"\n  {'─'*56}")
    print(f"  {flag}")
    print(f"  {'─'*56}")
    print(f"  Statement  : {result['statement']}")
    print(f"  Detection  : {result['detection']['h_prob']}% hallucination "
          f"(threshold: {threshold})")

    if result["nli"]:
        n = result["nli"]
        print(f"  NLI        : {n['label']} | Entailment: {n['entailment']}% "
              f"| Contradiction: {n['contradiction']}%")

    print(f"  Weighted   : {result['weighted']}% | Confidence: {result['confidence']}")

    if result["evidence"]:
        prev = result["evidence"][:120].replace("\n", " ")
        print(f"  Evidence   : [{result['ev_source']}] {prev}...")

    if result["corrected"]:
        print(f"\n  ✏ Corrected: {result['corrected']}")

    print()


# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":

    # Command-line mode
    if len(sys.argv) > 1:
        stmt = " ".join(sys.argv[1:])
        print(f"\nAnalyzing: {stmt}")
        display(full_pipeline(stmt))
        sys.exit(0)

    # Interactive mode
    print("=" * 58)
    print("  Hallucination Detector — Full Pipeline")
    print("  Detection works WITHOUT Wikipedia evidence.")
    print("  Wikipedia is used for NLI & correction only.")
    print("  Type 'quit' to exit.")
    print("=" * 58)

    while True:
        try:
            stmt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break

        if not stmt or stmt.lower() in ("quit","exit","q"):
            print("Exiting."); break

        display(full_pipeline(stmt))
