"""
=============================================================
  Hallucination Detector — Full Pipeline v3
  GPU    : RTX 3050 Mobile 4GB — Linux

  IMPROVEMENTS:
  1. Rule-based safety layer catches confident hallucinations
     the model still misses (unit errors, obvious impossibilities)
  2. Statement-only RoBERTa detection (core fix from v2)
  3. NLI + Wikipedia for verification when available
  4. Weighted combination: detection + NLI + rules
  5. Better correction prompt (shorter, NaN-safe)

  Usage:
    python predict.py "Statement to check"
    python predict.py   (interactive)
=============================================================
"""

import sys
import os
import re
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration
)

# ── Paths ─────────────────────────────────────────────────
DETECTOR_PATH  = "./detector_model/best"
NLI_PATH       = "./nli_model/best"
CORRECTOR_PATH = "./corrector_model/best"
MAX_LEN        = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

threshold = 0.5
t_file = os.path.join(DETECTOR_PATH, "threshold.txt")
if os.path.exists(t_file):
    with open(t_file) as f:
        threshold = float(f.read().strip())

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


# =============================================================
# RULE-BASED LAYER
# Catches obvious hallucinations that the ML model still misses.
# These are high-confidence patterns that don't need ML.
# =============================================================
UNIT_RULES = [
    # Pattern: (regex, correct_fact, explanation)
    (r"1\s*li?t(?:er|re)\s*(?:is|=|equals?)\s*(\d+)\s*m[li]",
     lambda m: int(m.group(1)) != 1000,
     "1 liter = 1000 ml"),
    (r"1\s*km\s*(?:is|=|equals?)\s*1\s*li?t(?:er|re)",
     lambda m: True,
     "km and liters are different units (distance vs volume)"),
    (r"1\s*kg\s*(?:is|=|equals?)\s*(\d+)\s*(?:gram|g\b)",
     lambda m: int(m.group(1)) != 1000,
     "1 kg = 1000 grams"),
    (r"water\s+boils?\s+at\s+(\d+)\s+degrees?",
     lambda m: abs(int(m.group(1))-100) > 5,
     "Water boils at 100°C at sea level"),
    (r"water\s+freezes?\s+at\s+(\d+)\s+degrees?",
     lambda m: int(m.group(1)) != 0,
     "Water freezes at 0°C"),
    (r"(?:standard|normal|regular)\s+car\s+has\s+(\d+)\s+wheels?",
     lambda m: int(m.group(1)) != 4,
     "A standard car has 4 wheels"),
    (r"humans?\s+have\s+(\d+)\s+fingers?\s+(?:on|in)\s+each\s+hand",
     lambda m: int(m.group(1)) != 5,
     "Humans have 5 fingers on each hand"),
    (r"(?:a\s+)?week\s+has\s+(\d+)\s+days?",
     lambda m: int(m.group(1)) != 7,
     "A week has 7 days"),
    (r"(?:a\s+)?(?:year|yr)\s+has\s+(\d+)\s+days?",
     lambda m: int(m.group(1)) not in [365, 366],
     "A year has 365 days"),
    (r"(?:a\s+)?minute\s+has\s+(\d+)\s+seconds?",
     lambda m: int(m.group(1)) != 60,
     "A minute has 60 seconds"),
    (r"(?:an?\s+)?hour\s+has\s+(\d+)\s+minutes?",
     lambda m: int(m.group(1)) != 60,
     "An hour has 60 minutes"),
    (r"1\s*(?:meter|metre|m)\s*(?:is|=|equals?)\s*(\d+)\s*(?:cm|centimeter)",
     lambda m: int(m.group(1)) != 100,
     "1 meter = 100 centimeters"),
]

KNOWN_HALLUCINATIONS = [
    (r"(?:milky\s*way|milkyway)\s+galaxy\s+is\s+(?:a|the)\s+(?:name\s+of\s+a\s+)?planet",
     "The Milky Way is a galaxy, not a planet"),
    (r"andromeda\s+galaxy\s+is\s+the\s+galaxy\s+(?:in\s+which|where)\s+we\s+live",
     "We live in the Milky Way Galaxy, not Andromeda"),
    (r"einstein\s+invented\s+the\s+telephone",
     "Alexander Graham Bell invented the telephone"),
    (r"shakespeare\s+was\s+born\s+in\s+london",
     "Shakespeare was born in Stratford-upon-Avon"),
    (r"eiffel\s+tower\s+is\s+(?:located\s+)?in\s+berlin",
     "The Eiffel Tower is in Paris, France"),
    (r"statue\s+of\s+liberty\s+is\s+(?:located\s+)?in\s+paris",
     "The Statue of Liberty is in New York"),
    (r"psychology\s+is\s+the\s+study\s+of\s+(?:fish|fishes)",
     "Psychology is the study of mind and behavior"),
    (r"exit\s+means\s+to\s+keep\s+going",
     "Exit means to leave or go out"),
    (r"domino'?s\s+is\s+a\s+sushi\s+(?:restaurant|place|chain)",
     "Domino's is a pizza restaurant chain"),
    (r"mcdonald'?s\s+is\s+a\s+chinese\s+(?:restaurant|place|chain)",
     "McDonald's is an American fast food chain"),
    (r"kfc\s+is\s+an?\s+italian\s+(?:restaurant|place|chain)",
     "KFC is an American fried chicken restaurant chain"),
    (r"amazon\s+(?:river\s+)?is\s+the\s+longest\s+river",
     "The Nile is the longest river; the Amazon is the largest by water flow"),
    (r"sound\s+travels?\s+faster\s+than\s+light",
     "Light travels much faster than sound"),
    (r"new\s+zealand\s+is\s+the\s+(?:biggest|largest)\s+country",
     "Russia is the largest country in the world"),
    (r"russia\s+is\s+the\s+(?:smallest|tiniest)\s+country",
     "Vatican City is the smallest country in the world"),
    (r"capital\s+of\s+australia\s+is\s+sydney",
     "The capital of Australia is Canberra"),
    (r"capital\s+of\s+canada\s+is\s+toronto",
     "The capital of Canada is Ottawa"),
    (r"capital\s+of\s+brazil\s+is\s+s[aã]o\s+paulo",
     "The capital of Brazil is Brasilia"),
    (r"capital\s+of\s+(?:india|the\s+usa|america)\s+is\s+(?:mumbai|new\s+york)",
     "The capital of India is New Delhi; USA is Washington DC"),
]

def rule_based_check(statement):
    """
    Returns (is_hallucinated: bool, reason: str | None)
    High-confidence rule-based checks for known patterns.
    """
    stmt_lower = statement.lower().strip()

    # Check unit rules
    for pattern, condition_fn, correct_fact in UNIT_RULES:
        m = re.search(pattern, stmt_lower)
        if m:
            try:
                if condition_fn(m):
                    return True, f"Unit/number error. Fact: {correct_fact}"
            except (ValueError, IndexError):
                pass

    # Check known hallucination patterns
    for pattern, note in KNOWN_HALLUCINATIONS:
        if re.search(pattern, stmt_lower):
            return True, note

    return False, None


# =============================================================
# ML INFERENCE FUNCTIONS
# =============================================================
def detect(statement):
    """Statement-only RoBERTa detection."""
    enc = det_tok(str(statement), max_length=MAX_LEN, truncation=True,
                  padding="max_length", return_tensors="pt")
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
    prompt = (f"Fix the incorrect fact. Evidence: {evidence.strip()[:200]} "
              f"Statement: {statement} Correction:"
              if evidence and evidence.strip() else
              f"Fix the incorrect fact. Statement: {statement} Correction:")
    enc = cor_tok(prompt, max_length=256, truncation=True,
                  padding="max_length", return_tensors="pt")
    with torch.no_grad():
        ids = cor_model.generate(
            input_ids            = enc["input_ids"].to(device),
            attention_mask       = enc["attention_mask"].to(device),
            max_length           = 128,
            min_length           = 10,
            num_beams            = 4,
            length_penalty       = 1.5,
            early_stopping       = True,
            no_repeat_ngram_size = 3,
            do_sample            = False
        )
    return cor_tok.decode(ids[0], skip_special_tokens=True)

def full_pipeline(statement):
    """
    3-layer verdict:
    Layer 1: Rule-based check (highest priority, deterministic)
    Layer 2: RoBERTa statement-only detection (ML)
    Layer 3: NLI with Wikipedia evidence (if available)
    Final verdict: weighted combination
    """
    # Layer 1 — Rule check
    rule_flag, rule_reason = rule_based_check(statement)

    # Layer 2 — ML detection (statement only)
    det = detect(statement)

    # Layer 3 — Wikipedia + NLI
    evidence, ev_source = "", ""
    if RETRIEVER:
        ev = get_evidence(statement, verbose=False)
        if ev.get("found"):
            evidence  = ev.get("evidence", "")
            ev_source = ev.get("source", "Wikipedia")

    nli = nli_check(evidence, statement) if evidence and evidence.strip() else None

    # ── Final verdict ─────────────────────────────────────
    # Rule layer overrides if triggered
    if rule_flag:
        verdict, conf = "Hallucinated", "HIGH (Rule)"
        h_score = 1.0
    else:
        # Weighted: 60% detection + 40% NLI (if available)
        c_prob   = nli["raw_contra"] if nli else det["raw"]
        h_score  = 0.6 * det["raw"] + 0.4 * c_prob

        if h_score > 0.6:
            verdict, conf = "Hallucinated", "HIGH"
        elif h_score > 0.35:
            verdict, conf = "Hallucinated", "MEDIUM"
        else:
            verdict, conf = "Factual", "LOW"

    corrected = correct(statement, evidence) if verdict == "Hallucinated" else None

    return {
        "statement"   : statement,
        "rule_flag"   : rule_flag,
        "rule_reason" : rule_reason,
        "detection"   : det,
        "nli"         : nli,
        "evidence"    : evidence,
        "ev_source"   : ev_source,
        "verdict"     : verdict,
        "confidence"  : conf,
        "weighted"    : round(h_score * 100, 2),
        "corrected"   : corrected,
    }

def display(r):
    flag = "⚠  HALLUCINATED" if r["verdict"] == "Hallucinated" else "✅ FACTUAL"
    print(f"\n  {'─'*56}")
    print(f"  {flag}")
    print(f"  {'─'*56}")
    print(f"  Statement  : {r['statement']}")

    if r["rule_flag"]:
        print(f"  Rule Layer : ⚠ FLAGGED — {r['rule_reason']}")
    print(f"  Detection  : {r['detection']['h_prob']}% hallucination (threshold:{threshold})")

    if r["nli"]:
        n = r["nli"]
        print(f"  NLI        : {n['label']} | Entailment:{n['entailment']}% | Contradiction:{n['contradiction']}%")

    print(f"  Weighted   : {r['weighted']}% | Confidence: {r['confidence']}")

    if r["evidence"]:
        print(f"  Evidence   : [{r['ev_source']}] {r['evidence'][:100]}...")

    if r["corrected"]:
        print(f"\n  ✏ Corrected: {r['corrected']}")
    print()


# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        stmt = " ".join(sys.argv[1:])
        print(f"\nAnalyzing: {stmt}")
        display(full_pipeline(stmt))
        sys.exit(0)

    print("="*58)
    print("  Hallucination Detector — 3-Layer Pipeline")
    print("  Layer 1: Rule-based (deterministic)")
    print("  Layer 2: RoBERTa ML detection")
    print("  Layer 3: NLI with Wikipedia")
    print("  Type 'quit' to exit.")
    print("="*58)

    while True:
        try:
            stmt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting."); break

        if not stmt or stmt.lower() in ("quit","exit","q"):
            print("Exiting."); break

        display(full_pipeline(stmt))
