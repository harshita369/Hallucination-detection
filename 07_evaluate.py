"""
=============================================================
  Phase 7 — Full Pipeline Evaluation
  GPU    : RTX 3050 Mobile 4GB — Linux

  FIXES:
  1. Retriever import fixed with sys.path
  2. All 790 TruthfulQA samples evaluated
  3. Threshold loaded from threshold.txt
  4. Detection runs statement-only (no evidence needed)
  5. NLI only used when Wikipedia evidence available
  6. Strict + partial TruthfulQA accuracy reported
  7. Per-source dataset accuracy breakdown
=============================================================
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# Fix retriever import path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration
)
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)

try:
    from evaluate import load as load_metric
    rouge_metric = load_metric("rouge")
    ROUGE_OK = True
except:
    ROUGE_OK = False

try:
    from retriever import get_evidence
    RETRIEVER = True
    print("✅ Retriever loaded")
except Exception as e:
    RETRIEVER = False
    print(f"⚠ Retriever not available: {e}")
    def get_evidence(claim, verbose=False):
        return {"evidence": "", "found": False}

CONFIG = {
    "detector_path"  : "./detector_model/best",
    "nli_path"       : "./nli_model/best",
    "corrector_path" : "./corrector_model/best",
    "test_path"      : "./outputs/detection_test.csv",
    "benchmark_path" : "./outputs/benchmark_truthfulqa.csv",
    "output_dir"     : "./evaluation_results",
    "max_length"     : 128,
    "max_input_length": 256,
    "max_output_length": 128,
    "benchmark_samples": None,   # all 790
    "batch_size"     : 16,
    "seed"           : 42,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

print("=" * 60)
print("  Phase 7 — Full Pipeline Evaluation")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"   Device: {device}\n")

# ── Load models ───────────────────────────────────────────
print("--- Loading models ---")

det_tok   = AutoTokenizer.from_pretrained(CONFIG["detector_path"])
det_model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["detector_path"]).to(device)
det_model.eval()
print("   ✅ Detection model")

nli_tok   = AutoTokenizer.from_pretrained(CONFIG["nli_path"])
nli_model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["nli_path"]).to(device)
nli_model.eval()
print("   ✅ NLI model")

cor_tok   = T5Tokenizer.from_pretrained(CONFIG["corrector_path"], legacy=False)
cor_model = T5ForConditionalGeneration.from_pretrained(
    CONFIG["corrector_path"]).to(device)
cor_model.eval()
print("   ✅ Correction model")

# Load threshold
threshold = 0.5
t_file = os.path.join(CONFIG["detector_path"], "threshold.txt")
if os.path.exists(t_file):
    with open(t_file) as f:
        threshold = float(f.read().strip())
    print(f"   ✅ Threshold loaded: {threshold}")
else:
    print(f"   ⚠ threshold.txt not found, using 0.5")
print()

# ── Inference functions ───────────────────────────────────
def detect(statement):
    """Statement-only detection — no evidence needed."""
    enc = det_tok(str(statement), max_length=CONFIG["max_length"],
                  truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        out = det_model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
    probs  = torch.softmax(out.logits, dim=1)[0]
    h_prob = probs[1].item()
    return {
        "label"     : 1 if h_prob > threshold else 0,
        "label_text": "Hallucinated" if h_prob > threshold else "Factual",
        "h_prob"    : round(h_prob * 100, 2),
        "f_prob"    : round(probs[0].item() * 100, 2),
        "raw"       : h_prob
    }

def nli_check(premise, hypothesis):
    enc = nli_tok(str(premise), str(hypothesis), max_length=CONFIG["max_length"],
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
    enc = cor_tok(prompt, max_length=CONFIG["max_input_length"],
                  truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        ids = cor_model.generate(
            input_ids            = enc["input_ids"].to(device),
            attention_mask       = enc["attention_mask"].to(device),
            max_length           = CONFIG["max_output_length"],
            min_length           = 15,
            num_beams            = 4,
            length_penalty       = 2.0,
            early_stopping       = True,
            no_repeat_ngram_size = 3
        )
    return cor_tok.decode(ids[0], skip_special_tokens=True)

def pipeline(statement, evidence=""):
    det = detect(statement)
    nli = nli_check(evidence, statement) if evidence and evidence.strip() else None

    # Weighted combination
    c_prob = nli["raw_contra"] if nli else det["raw"]
    weighted = 0.6 * det["raw"] + 0.4 * c_prob

    if weighted > 0.6:
        verdict, conf = "Hallucinated", "HIGH"
    elif weighted > 0.35:
        verdict, conf = "Hallucinated", "MEDIUM"
    else:
        verdict, conf = "Factual", "LOW"

    corrected = correct(statement, evidence) if verdict == "Hallucinated" else None

    return {
        "statement" : statement,
        "detection" : det,
        "nli"       : nli,
        "verdict"   : verdict,
        "confidence": conf,
        "weighted"  : round(weighted*100, 2),
        "corrected" : corrected,
    }

# =============================================================
# PART 1: DETECTION EVALUATION
# =============================================================
print("=" * 60)
print("  PART 1: Detection Model Evaluation")
print("=" * 60)

df_test = pd.read_csv(CONFIG["test_path"], encoding="latin-1")
df_test = df_test.dropna(subset=["statement","label"]).reset_index(drop=True)
df_test["statement"] = df_test["statement"].astype(str)
print(f"\n   Test samples: {len(df_test)} | Threshold: {threshold}\n")

all_preds, all_labels, all_probs = [], [], []
for i in range(0, len(df_test), CONFIG["batch_size"]):
    batch = df_test.iloc[i:i+CONFIG["batch_size"]]
    for _, row in batch.iterrows():
        r = detect(str(row["statement"]))
        all_preds.append(r["label"])
        all_labels.append(int(row["label"]))
        all_probs.append(r["raw"])
    if (i // CONFIG["batch_size"] + 1) % 50 == 0:
        done = min(i+CONFIG["batch_size"], len(df_test))
        print(f"   Processed {done}/{len(df_test)}...")

acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="macro")
rec  = recall_score(all_labels, all_preds, average="macro")
f1   = f1_score(all_labels, all_preds, average="macro")
rep  = classification_report(all_labels, all_preds,
                              target_names=["Factual","Hallucinated"])
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc     = auc(fpr, tpr)

print(f"\n   Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
print(f"   Precision : {prec:.4f}")
print(f"   Recall    : {rec:.4f}")
print(f"   Macro F1  : {f1:.4f}")
print(f"   ROC-AUC   : {roc_auc:.4f}")
print(f"\n{rep}")

if "source_dataset" in df_test.columns:
    print("   Per-source accuracy:")
    for src in df_test["source_dataset"].unique():
        idx  = df_test[df_test["source_dataset"] == src].index.tolist()
        s_l  = [all_labels[i] for i in idx]
        s_p  = [all_preds[i]  for i in idx]
        s_a  = accuracy_score(s_l, s_p)
        print(f"     {src:<35}: {s_a:.3f}")

# =============================================================
# PART 2: TRUTHFULQA
# =============================================================
print("\n" + "="*60)
print("  PART 2: TruthfulQA Benchmark")
print("="*60)

df_bench = pd.read_csv(CONFIG["benchmark_path"], encoding="latin-1")
if CONFIG["benchmark_samples"]:
    df_bench = df_bench.sample(n=min(CONFIG["benchmark_samples"], len(df_bench)),
                                random_state=CONFIG["seed"]).reset_index(drop=True)

print(f"\n   Samples   : {len(df_bench)}")
print(f"   Categories: {df_bench['category'].nunique()}")
print(f"   Wikipedia : {'enabled' if RETRIEVER else 'disabled'}\n")

bench_rows = []
correct_both, correct_either = 0, 0
corr_texts, ref_texts = [], []

for i, row in df_bench.iterrows():
    question      = str(row["question"])
    correct_ans   = str(row["correct_answer"])
    incorrect_ans = str(row["incorrect_answer"])
    category      = str(row["category"])

    evidence = ""
    if RETRIEVER:
        ev = get_evidence(question, verbose=False)
        evidence = ev.get("evidence", "")

    cr = detect(correct_ans)
    ir = detect(incorrect_ans)

    both_ok = (cr["label"] == 0 and ir["label"] == 1)
    if both_ok: correct_both += 1
    if cr["label"] == 0 or ir["label"] == 1: correct_either += 1

    corrected = correct(incorrect_ans, evidence)
    corr_texts.append(corrected)
    ref_texts.append(correct_ans)

    bench_rows.append({
        "question"      : question,
        "correct_answer": correct_ans,
        "incorrect_ans" : incorrect_ans,
        "category"      : category,
        "correct_pred"  : cr["label"],
        "incorrect_pred": ir["label"],
        "both_correct"  : both_ok,
        "corrected"     : corrected,
        "evidence_found": bool(evidence),
    })

    if (i+1) % 50 == 0:
        print(f"   Processed {i+1}/{len(df_bench)}...")

df_res = pd.DataFrame(bench_rows)
strict  = correct_both  / len(df_bench)
partial = correct_either / len(df_bench)

print(f"\n   TruthfulQA Strict Accuracy  : {strict:.4f} ({strict*100:.2f}%)")
print(f"   TruthfulQA Partial Accuracy : {partial:.4f} ({partial*100:.2f}%)")
print(f"   Evidence retrieved          : {df_res['evidence_found'].sum()}/{len(df_bench)}")

cat_acc = df_res.groupby("category")["both_correct"].mean().sort_values(ascending=False)
print(f"\n   Per-category (top 10):")
for cat, val in cat_acc.head(10).items():
    print(f"     {cat:<35}: {val:.2f}")

# =============================================================
# PART 3: ROUGE
# =============================================================
print("\n" + "="*60)
print("  PART 3: Correction ROUGE Scores")
print("="*60)

rouge_scores = {}
if ROUGE_OK and corr_texts:
    r = rouge_metric.compute(predictions=corr_texts, references=ref_texts)
    rouge_scores = {
        "ROUGE-1": round(r["rouge1"]*100, 2),
        "ROUGE-2": round(r["rouge2"]*100, 2),
        "ROUGE-L": round(r["rougeL"]*100, 2),
    }
    for k, v in rouge_scores.items():
        print(f"\n   {k}: {v}")
    print("\n   (Score > 30 is good for correction tasks)")

# =============================================================
# SAVE
# =============================================================
df_res.to_csv(os.path.join(CONFIG["output_dir"], "pipeline_results.csv"), index=False)

with open(os.path.join(CONFIG["output_dir"], "full_report.txt"), "w") as f:
    f.write("HALLUCINATION DETECTION — IMPROVED EVALUATION\n" + "="*60 + "\n\n")
    f.write(f"Threshold : {threshold}\n\n")
    f.write("DETECTION MODEL\n" + "-"*40 + "\n")
    f.write(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)\n")
    f.write(f"Macro F1  : {f1:.4f}\n")
    f.write(f"ROC-AUC   : {roc_auc:.4f}\n\n")
    f.write(rep + "\n\n")
    f.write("TRUTHFULQA\n" + "-"*40 + "\n")
    f.write(f"Strict Accuracy : {strict:.4f}\n")
    f.write(f"Partial Accuracy: {partial:.4f}\n\n")
    for cat, val in cat_acc.items():
        f.write(f"  {cat:<35}: {val:.2f}\n")
    if rouge_scores:
        f.write("\nROUGE\n" + "-"*40 + "\n")
        for k, v in rouge_scores.items():
            f.write(f"  {k}: {v}\n")
print("\n✅ Reports saved")

# =============================================================
# PLOTS
# =============================================================
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0,0])
cm  = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=["Factual","Hallucinated"]).plot(
    ax=ax1, colorbar=False, cmap="Blues")
ax1.set_title("Detection\nConfusion Matrix", fontweight="bold", fontsize=12)

ax2 = fig.add_subplot(gs[0,1])
ax2.plot(fpr, tpr, "darkorange", lw=2, label=f"AUC={roc_auc:.3f}")
ax2.plot([0,1],[0,1],"navy",lw=1,linestyle="--")
ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
ax2.set_title("ROC Curve", fontweight="bold", fontsize=12)
ax2.legend(loc="lower right"); ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0,2])
metrics = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
values  = [acc, prec, rec, f1, roc_auc]
colors  = ["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336"]
bars = ax3.bar(metrics, values, color=colors, alpha=0.85)
ax3.set_ylim(0, 1.15); ax3.set_title("Metrics", fontweight="bold", fontsize=12)
ax3.grid(True, axis="y", alpha=0.3)
for b, v in zip(bars, values):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
             f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

ax4 = fig.add_subplot(gs[1,:2])
top15 = cat_acc.head(15)
col4  = ["#4CAF50" if v>=0.6 else "#FF9800" if v>=0.4 else "#F44336" for v in top15.values]
ax4.barh(top15.index, top15.values, color=col4, alpha=0.85)
ax4.set_xlabel("Accuracy")
ax4.set_title("TruthfulQA Per-Category Accuracy", fontweight="bold", fontsize=12)
ax4.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="50% baseline")
ax4.legend(); ax4.grid(True, axis="x", alpha=0.3)
for val, patch in zip(top15.values, ax4.patches):
    ax4.text(val+0.01, patch.get_y()+patch.get_height()/2,
             f"{val:.2f}", va="center", fontsize=8)

ax5 = fig.add_subplot(gs[1,2])
if rouge_scores:
    rn = list(rouge_scores.keys())
    rv = [v/100 for v in rouge_scores.values()]
    b5 = ax5.bar(rn, rv, color=["#3F51B5","#009688","#FF5722"], alpha=0.85)
    ax5.set_ylim(0, 1.0)
    ax5.axhline(y=0.3, color="green", linestyle="--", label="Good (0.30)")
    ax5.set_title("ROUGE Scores", fontweight="bold", fontsize=12)
    ax5.legend(); ax5.grid(True, axis="y", alpha=0.3)
    for b, v in zip(b5, rv):
        ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                 f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

fig.suptitle("Hallucination Detection & Correction Engine — Full Evaluation",
             fontsize=14, fontweight="bold")
plt.savefig(os.path.join(CONFIG["output_dir"], "full_evaluation.png"),
            dpi=150, bbox_inches="tight")
print("✅ Plots saved")

# =============================================================
# PIPELINE DEMO
# =============================================================
print("\n" + "="*60)
print("  PART 4: Full Pipeline Demo")
print("="*60)

demo_cases = [
    {"s": "Albert Einstein invented the telephone in 1876.",
     "e": "Alexander Graham Bell patented the telephone in 1876."},
    {"s": "The Eiffel Tower is in Berlin, Germany.",
     "e": "The Eiffel Tower is on the Champ de Mars in Paris, France."},
    {"s": "Water boils at 100 degrees Celsius at sea level.",
     "e": "Water boils at 100 degrees Celsius at standard atmospheric pressure."},
    {"s": "The Amazon is the longest river in the world.",
     "e": "The Nile at 6,650 km is the longest river. The Amazon is largest by flow."},
    {"s": "Shakespeare was born in London.",
     "e": "Shakespeare was born in Stratford-upon-Avon in April 1564."},
]

for i, case in enumerate(demo_cases, 1):
    r = pipeline(case["s"], case["e"])
    flag = "⚠" if r["verdict"] == "Hallucinated" else "✅"
    print(f"\n  {'─'*54}")
    print(f"  {i}. {flag} {r['verdict'].upper()} (weighted: {r['weighted']}%)")
    print(f"  Statement : {case['s']}")
    print(f"  Detection : {r['detection']['h_prob']}%")
    if r["nli"]:
        print(f"  NLI       : {r['nli']['label']} ({r['nli']['contradiction']}% contradiction)")
    print(f"  Confidence: {r['confidence']}")
    if r["corrected"]:
        print(f"  Corrected : {r['corrected']}")

print(f"\n{'='*60}")
print(f"  EVALUATION COMPLETE")
print(f"{'='*60}")
print(f"\n  Detection: Acc={acc*100:.2f}% | F1={f1:.4f} | AUC={roc_auc:.4f}")
print(f"  TruthfulQA: Strict={strict*100:.2f}% | Partial={partial*100:.2f}%")
if rouge_scores:
    print(f"  Correction: ROUGE-L={rouge_scores['ROUGE-L']}")
print(f"\n  Output: {os.path.abspath(CONFIG['output_dir'])}/")
print("="*60)
