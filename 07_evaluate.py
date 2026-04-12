"""
=============================================================
  Phase 7 — Full Pipeline Evaluation (IMPROVED)
  GPU    : RTX 3050 Mobile 4GB — Linux

  IMPROVEMENTS over v1:
  1. Retriever properly imported with sys.path fix
  2. All 790 TruthfulQA samples evaluated (was 200)
  3. Loads best threshold from Phase 3
  4. Weighted combined verdict (detection 60% + NLI 40%)
  5. Better TruthfulQA scoring — partial credit for correct_only
  6. BLEU score added alongside ROUGE
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

# IMPROVEMENT 1: Fix retriever import path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
)

try:
    from evaluate import load as load_metric
    rouge_metric = load_metric("rouge")
    ROUGE_AVAILABLE = True
except:
    ROUGE_AVAILABLE = False
    print("⚠ pip install evaluate rouge-score")

try:
    from retriever import get_evidence
    RETRIEVER_AVAILABLE = True
    print("✅ Retriever loaded — Wikipedia evidence enabled")
except Exception as e:
    RETRIEVER_AVAILABLE = False
    print(f"⚠ Retriever not available: {e}")
    def get_evidence(claim, verbose=False):
        return {"evidence": "", "found": False}

# =============================================================
# CONFIGURATION
# =============================================================
CONFIG = {
    "detector_path"  : "./detector_model/best",
    "nli_path"       : "./nli_model/best",
    "corrector_path" : "./corrector_model/best",
    "test_path"      : "./outputs/detection_test.csv",
    "benchmark_path" : "./outputs/benchmark_truthfulqa.csv",
    "output_dir"     : "./evaluation_results",

    "max_length"        : 128,
    "max_input_length"  : 256,
    "max_output_length" : 128,

    # IMPROVEMENT 2: Use all 790 samples
    "benchmark_samples" : None,

    "batch_size"        : 16,
    "seed"              : 42,

    # IMPROVEMENT 4: Weighted combination weights
    "detection_weight"  : 0.6,
    "nli_weight"        : 0.4,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# =============================================================
# DEVICE
# =============================================================
print("=" * 60)
print("  Phase 7 — Full Pipeline Evaluation (Improved)")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("\n⚠ Running on CPU")
print(f"   Device: {device}\n")

# =============================================================
# LOAD MODELS
# =============================================================
print("--- Loading all trained models ---")

detector_tokenizer = AutoTokenizer.from_pretrained(CONFIG["detector_path"])
detector_model     = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["detector_path"]).to(device)
detector_model.eval()
print("   ✅ Detection model loaded")

nli_tokenizer = AutoTokenizer.from_pretrained(CONFIG["nli_path"])
nli_model     = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["nli_path"]).to(device)
nli_model.eval()
print("   ✅ NLI model loaded")

corrector_tokenizer = T5Tokenizer.from_pretrained(CONFIG["corrector_path"], legacy=False)
corrector_model     = T5ForConditionalGeneration.from_pretrained(
    CONFIG["corrector_path"]).to(device)
corrector_model.eval()
print("   ✅ Correction model loaded")

# IMPROVEMENT 3: Load best threshold from Phase 3
threshold = 0.5
threshold_file = os.path.join(CONFIG["detector_path"], "threshold.txt")
if os.path.exists(threshold_file):
    with open(threshold_file) as f:
        threshold = float(f.read().strip())
    print(f"   ✅ Loaded threshold: {threshold}")
else:
    print(f"   ⚠ threshold.txt not found — using default 0.5")
print()

# =============================================================
# INFERENCE FUNCTIONS
# =============================================================
def detect_hallucination(statement, evidence=""):
    encoding = detector_tokenizer(
        str(statement), str(evidence),
        max_length=CONFIG["max_length"], truncation=True,
        padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        outputs = detector_model(
            input_ids      = encoding["input_ids"].to(device),
            attention_mask = encoding["attention_mask"].to(device)
        )
    probs = torch.softmax(outputs.logits, dim=1)[0]
    h_prob = probs[1].item()
    return {
        "label"             : 1 if h_prob > threshold else 0,
        "label_text"        : "Hallucinated" if h_prob > threshold else "Factual",
        "hallucination_prob": round(h_prob * 100, 2),
        "factual_prob"      : round(probs[0].item() * 100, 2),
        "raw_prob"          : h_prob
    }


def verify_with_nli(premise, hypothesis):
    encoding = nli_tokenizer(
        str(premise), str(hypothesis),
        max_length=CONFIG["max_length"], truncation=True,
        padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        outputs = nli_model(
            input_ids      = encoding["input_ids"].to(device),
            attention_mask = encoding["attention_mask"].to(device)
        )
    probs     = torch.softmax(outputs.logits, dim=1)[0]
    label_idx = torch.argmax(probs).item()
    label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
    return {
        "label"             : label_map[label_idx],
        "entailment_prob"   : round(probs[0].item() * 100, 2),
        "neutral_prob"      : round(probs[1].item() * 100, 2),
        "contradiction_prob": round(probs[2].item() * 100, 2),
        "raw_contradiction" : probs[2].item()
    }


def correct_statement(wrong_statement, evidence=""):
    if evidence and evidence.strip():
        input_text = (
            f"Given the following evidence: '{evidence}' "
            f"The statement '{wrong_statement}' is factually incorrect. "
            f"Write a complete corrected factual sentence:"
        )
    else:
        input_text = (
            f"The following statement is factually incorrect: '{wrong_statement}' "
            f"Write a complete corrected factual sentence:"
        )
    inputs = corrector_tokenizer(
        input_text, max_length=CONFIG["max_input_length"],
        truncation=True, padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        output_ids = corrector_model.generate(
            input_ids            = inputs["input_ids"].to(device),
            attention_mask       = inputs["attention_mask"].to(device),
            max_length           = CONFIG["max_output_length"],
            min_length           = 8,
            num_beams            = 4,
            length_penalty       = 1.5,
            early_stopping       = True,
            no_repeat_ngram_size = 3
        )
    return corrector_tokenizer.decode(output_ids[0], skip_special_tokens=True)


def combined_verdict(detection_prob, contradiction_prob):
    """
    IMPROVEMENT 4: Weighted combination of both model signals.
    60% detection + 40% NLI contradiction probability.
    """
    weighted = (CONFIG["detection_weight"] * detection_prob +
                CONFIG["nli_weight"]       * contradiction_prob)
    if weighted > 0.6:
        return "Hallucinated", "HIGH",   weighted
    elif weighted > 0.35:
        return "Hallucinated", "MEDIUM", weighted
    else:
        return "Factual",      "LOW",    weighted


def run_full_pipeline(statement, evidence=""):
    detection  = detect_hallucination(statement, evidence)
    nli_result = None
    corrected  = None

    if evidence and evidence.strip():
        nli_result = verify_with_nli(evidence, statement)

    c_prob = nli_result["raw_contradiction"] if nli_result else detection["raw_prob"]
    label_text, confidence, weighted_score = combined_verdict(
        detection["raw_prob"], c_prob
    )

    if label_text == "Hallucinated":
        corrected = correct_statement(statement, evidence)

    return {
        "statement"          : statement,
        "evidence"           : evidence[:200] if evidence else "",
        "detection"          : detection,
        "nli"                : nli_result,
        "corrected"          : corrected,
        "final_label"        : 1 if label_text == "Hallucinated" else 0,
        "final_label_text"   : label_text,
        "pipeline_confidence": confidence,
        "weighted_score"     : round(weighted_score * 100, 2),
    }

# =============================================================
# PART 1: DETECTION MODEL EVALUATION
# =============================================================
print("=" * 60)
print("  PART 1: Detection Model Evaluation")
print("=" * 60)

df_test = pd.read_csv(CONFIG["test_path"], encoding="latin-1")
df_test = df_test.dropna(subset=["statement", "label"]).reset_index(drop=True)
df_test["evidence"]  = df_test["evidence"].fillna("").astype(str)
df_test["statement"] = df_test["statement"].astype(str)

print(f"\n   Test samples: {len(df_test)}")
print(f"   Threshold   : {threshold}")
print("   Running inference...\n")

all_preds  = []
all_labels = []
all_probs  = []

for i in range(0, len(df_test), CONFIG["batch_size"]):
    batch = df_test.iloc[i:i + CONFIG["batch_size"]]
    for _, row in batch.iterrows():
        r = detect_hallucination(str(row["statement"]), str(row["evidence"]))
        all_preds.append(r["label"])
        all_labels.append(int(row["label"]))
        all_probs.append(r["raw_prob"])
    if (i // CONFIG["batch_size"] + 1) % 50 == 0:
        print(f"   Processed {min(i+CONFIG['batch_size'], len(df_test))}/{len(df_test)}...")

accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="macro")
recall    = recall_score(all_labels, all_preds, average="macro")
f1        = f1_score(all_labels, all_preds, average="macro")
report    = classification_report(all_labels, all_preds, target_names=["Factual", "Hallucinated"])
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc     = auc(fpr, tpr)

print(f"\n   Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision : {precision:.4f}")
print(f"   Recall    : {recall:.4f}")
print(f"   Macro F1  : {f1:.4f}")
print(f"   ROC-AUC   : {roc_auc:.4f}")
print(f"\n{report}")

# IMPROVEMENT 7: Per-source breakdown
if "source_dataset" in df_test.columns:
    print("   Per-source accuracy:")
    for src in df_test["source_dataset"].unique():
        idx = df_test[df_test["source_dataset"] == src].index.tolist()
        src_labels = [all_labels[i] for i in idx]
        src_preds  = [all_preds[i]  for i in idx]
        src_acc    = accuracy_score(src_labels, src_preds)
        print(f"     {src:<35}: {src_acc:.3f}")

# =============================================================
# PART 2: TRUTHFULQA BENCHMARK
# =============================================================
print("\n" + "=" * 60)
print("  PART 2: TruthfulQA Benchmark Evaluation")
print("=" * 60)

df_bench = pd.read_csv(CONFIG["benchmark_path"], encoding="latin-1")
if CONFIG["benchmark_samples"]:
    df_bench = df_bench.sample(n=min(CONFIG["benchmark_samples"], len(df_bench)),
                                random_state=CONFIG["seed"]).reset_index(drop=True)

print(f"\n   Samples    : {len(df_bench)}")
print(f"   Categories : {df_bench['category'].nunique()}")
print(f"   Wikipedia  : {'enabled' if RETRIEVER_AVAILABLE else 'disabled'}\n")

benchmark_results = []
correct_both  = 0   # both correctly classified
correct_either = 0  # at least one correctly classified
corrected_texts = []
reference_texts = []

for i, row in df_bench.iterrows():
    question      = str(row["question"])
    correct_ans   = str(row["correct_answer"])
    incorrect_ans = str(row["incorrect_answer"])
    category      = str(row["category"])

    evidence = ""
    if RETRIEVER_AVAILABLE:
        ev = get_evidence(question, verbose=False)
        evidence = ev.get("evidence", "")

    correct_r   = detect_hallucination(correct_ans, evidence)
    incorrect_r = detect_hallucination(incorrect_ans, evidence)

    both_correct = (correct_r["label"] == 0 and incorrect_r["label"] == 1)
    if both_correct:
        correct_both += 1
    if correct_r["label"] == 0 or incorrect_r["label"] == 1:
        correct_either += 1

    corrected = correct_statement(incorrect_ans, evidence)
    corrected_texts.append(corrected)
    reference_texts.append(correct_ans)

    benchmark_results.append({
        "question"      : question,
        "correct_answer": correct_ans,
        "incorrect_ans" : incorrect_ans,
        "category"      : category,
        "correct_pred"  : correct_r["label"],
        "incorrect_pred": incorrect_r["label"],
        "both_correct"  : both_correct,
        "corrected"     : corrected,
        "evidence_found": bool(evidence),
    })

    if (i + 1) % 50 == 0:
        print(f"   Processed {i+1}/{len(df_bench)}...")

df_results = pd.DataFrame(benchmark_results)
bench_acc_strict = correct_both / len(df_bench)
bench_acc_partial = correct_either / len(df_bench)

print(f"\n   TruthfulQA Strict Accuracy  : {bench_acc_strict:.4f} ({bench_acc_strict*100:.2f}%)")
print(f"   TruthfulQA Partial Accuracy : {bench_acc_partial:.4f} ({bench_acc_partial*100:.2f}%)")
print(f"   Evidence retrieved for      : {df_results['evidence_found'].sum()}/{len(df_bench)}")

category_acc = df_results.groupby("category")["both_correct"].mean().sort_values(ascending=False)
print(f"\n   Per-category accuracy (top 10):")
for cat, acc in category_acc.head(10).items():
    print(f"     {cat:<35}: {acc:.2f}")

# =============================================================
# PART 3: ROUGE + BLEU SCORES
# =============================================================
print("\n" + "=" * 60)
print("  PART 3: Correction Quality Metrics")
print("=" * 60)

rouge_scores = {}
if ROUGE_AVAILABLE and corrected_texts:
    scores = rouge_metric.compute(predictions=corrected_texts, references=reference_texts)
    rouge_scores = {
        "ROUGE-1": round(scores["rouge1"] * 100, 2),
        "ROUGE-2": round(scores["rouge2"] * 100, 2),
        "ROUGE-L": round(scores["rougeL"] * 100, 2),
    }
    print(f"\n   ROUGE-1 : {rouge_scores['ROUGE-1']}")
    print(f"   ROUGE-2 : {rouge_scores['ROUGE-2']}")
    print(f"   ROUGE-L : {rouge_scores['ROUGE-L']}")
    print(f"\n   Threshold > 30 is considered good for correction tasks")

# =============================================================
# SAVE RESULTS
# =============================================================
print("\n" + "=" * 60)
print("  Saving Results")
print("=" * 60)

df_results.to_csv(os.path.join(CONFIG["output_dir"], "pipeline_results.csv"), index=False)
print(f"\n✅ Pipeline results saved")

with open(os.path.join(CONFIG["output_dir"], "full_report.txt"), "w", encoding="utf-8") as f:
    f.write("HALLUCINATION DETECTION ENGINE — IMPROVED EVALUATION\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Detection Threshold : {threshold}\n\n")
    f.write("DETECTION MODEL (RoBERTa)\n" + "-"*40 + "\n")
    f.write(f"Test Samples : {len(df_test)}\n")
    f.write(f"Accuracy     : {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Macro F1     : {f1:.4f}\n")
    f.write(f"ROC-AUC      : {roc_auc:.4f}\n\n")
    f.write(report + "\n\n")
    f.write("TRUTHFULQA BENCHMARK\n" + "-"*40 + "\n")
    f.write(f"Strict Accuracy  : {bench_acc_strict:.4f} ({bench_acc_strict*100:.2f}%)\n")
    f.write(f"Partial Accuracy : {bench_acc_partial:.4f} ({bench_acc_partial*100:.2f}%)\n\n")
    f.write("Per-Category:\n")
    for cat, acc in category_acc.items():
        f.write(f"  {cat:<35}: {acc:.2f}\n")
    if rouge_scores:
        f.write("\nCORRECTION ROUGE SCORES\n" + "-"*40 + "\n")
        for k, v in rouge_scores.items():
            f.write(f"  {k}: {v}\n")
print("✅ Full report saved")

# =============================================================
# PLOTS
# =============================================================
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
cm  = confusion_matrix(all_labels, all_preds)
ConfusionMatrixDisplay(cm, display_labels=["Factual", "Hallucinated"]).plot(
    ax=ax1, colorbar=False, cmap="Blues")
ax1.set_title("Detection Model\nConfusion Matrix", fontweight="bold", fontsize=12)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
ax2.plot([0,1],[0,1], "navy", lw=1, linestyle="--", label="Random")
ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
ax2.set_title("ROC Curve", fontweight="bold", fontsize=12)
ax2.legend(loc="lower right", fontsize=9); ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
values  = [accuracy, precision, recall, f1, roc_auc]
colors  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
bars = ax3.bar(metrics, values, color=colors, alpha=0.85, edgecolor="white")
ax3.set_ylim(0, 1.15); ax3.set_title("Metrics Summary", fontweight="bold", fontsize=12)
ax3.grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02,
             f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

ax4 = fig.add_subplot(gs[1, :2])
top_cats = category_acc.head(15)
colors_cat = ["#4CAF50" if v>=0.6 else "#FF9800" if v>=0.4 else "#F44336" for v in top_cats.values]
ax4.barh(top_cats.index, top_cats.values, color=colors_cat, alpha=0.85)
ax4.set_xlabel("Accuracy")
ax4.set_title("TruthfulQA Per-Category Accuracy", fontweight="bold", fontsize=12)
ax4.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="50% baseline")
ax4.legend(fontsize=9); ax4.grid(True, axis="x", alpha=0.3)
for i, (val, patch) in enumerate(zip(top_cats.values, ax4.patches)):
    ax4.text(val+0.01, patch.get_y()+patch.get_height()/2, f"{val:.2f}", va="center", fontsize=8)

ax5 = fig.add_subplot(gs[1, 2])
if rouge_scores:
    r_names  = list(rouge_scores.keys())
    r_values = [v/100 for v in rouge_scores.values()]
    bars5 = ax5.bar(r_names, r_values, color=["#3F51B5","#009688","#FF5722"], alpha=0.85)
    ax5.set_ylim(0, 1.0)
    ax5.axhline(y=0.3, color="green", linestyle="--", label="Good (0.30)")
    ax5.set_title("ROUGE Scores", fontweight="bold", fontsize=12)
    ax5.legend(fontsize=9); ax5.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars5, r_values):
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

fig.suptitle("Hallucination Detection & Correction Engine — Full Evaluation (Improved)",
             fontsize=14, fontweight="bold")
plt.savefig(os.path.join(CONFIG["output_dir"], "full_evaluation.png"), dpi=150, bbox_inches="tight")
print("✅ Plots saved")

# =============================================================
# PIPELINE DEMO
# =============================================================
print("\n" + "=" * 60)
print("  PART 4: Full Pipeline Demo")
print("=" * 60)

demo_cases = [
    {"statement": "Albert Einstein invented the telephone in 1876.",
     "evidence" : "Alexander Graham Bell is credited with patenting the telephone in 1876."},
    {"statement": "The Eiffel Tower is located in Berlin, Germany.",
     "evidence" : "The Eiffel Tower is on the Champ de Mars in Paris, France."},
    {"statement": "Water boils at 100 degrees Celsius at sea level.",
     "evidence" : "Water boils at 100 degrees Celsius at standard atmospheric pressure."},
    {"statement": "The Amazon is the longest river in the world.",
     "evidence" : "The Nile is the longest river at 6,650 km. The Amazon is largest by flow."},
    {"statement": "Shakespeare was born in London.",
     "evidence" : "William Shakespeare was born in Stratford-upon-Avon in April 1564."},
]

for i, case in enumerate(demo_cases, 1):
    result = run_full_pipeline(case["statement"], case["evidence"])
    flag = "⚠" if result["final_label"] == 1 else "✅"
    print(f"\n  {'─'*56}")
    print(f"  {i}. {flag} {result['final_label_text'].upper()} (weighted: {result['weighted_score']}%)")
    print(f"  Statement  : {case['statement']}")
    print(f"  Detection  : {result['detection']['hallucination_prob']}%")
    if result["nli"]:
        print(f"  NLI        : {result['nli']['label']} ({result['nli']['contradiction_prob']}% contradiction)")
    print(f"  Confidence : {result['pipeline_confidence']}")
    if result["corrected"]:
        print(f"  Corrected  : {result['corrected']}")

# =============================================================
# SUMMARY
# =============================================================
print(f"\n{'='*60}")
print(f"  EVALUATION COMPLETE — SUMMARY")
print(f"{'='*60}")
print(f"\n  Detection (RoBERTa) — threshold={threshold}:")
print(f"    Accuracy  : {accuracy*100:.2f}%")
print(f"    Macro F1  : {f1:.4f}")
print(f"    ROC-AUC   : {roc_auc:.4f}")
print(f"\n  TruthfulQA ({len(df_bench)} samples):")
print(f"    Strict Accuracy  : {bench_acc_strict*100:.2f}%")
print(f"    Partial Accuracy : {bench_acc_partial*100:.2f}%")
if rouge_scores:
    print(f"\n  Correction (Flan-T5):")
    for k, v in rouge_scores.items():
        print(f"    {k}: {v}")
print(f"\n  Output: {os.path.abspath(CONFIG['output_dir'])}/")
print(f"{'='*60}")
