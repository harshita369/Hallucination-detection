"""
=============================================================
  Phase 7 — Full Pipeline Evaluation
  Purpose : Evaluate all trained models together on the
            TruthfulQA benchmark and test datasets
  Input   : detector_model/best/
            nli_model/best/
            corrector_model/best/
            outputs/detection_test.csv
            outputs/benchmark_truthfulqa.csv
  Output  : evaluation_results/
              full_report.txt
              confusion_matrix.png
              rouge_scores.png
              per_category_accuracy.png
              pipeline_results.csv
=============================================================
HOW TO RUN:
  Run AFTER Phase 3, 4, and 6 are all complete
  python evaluate.py
=============================================================
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

try:
    from evaluate import load as load_metric
    rouge  = load_metric("rouge")
    ROUGE_AVAILABLE = True
except:
    ROUGE_AVAILABLE = False
    print("⚠ Install evaluate library: pip install evaluate rouge-score")

# Import retriever from Phase 5
try:
    from retriever import get_evidence
    RETRIEVER_AVAILABLE = True
except:
    RETRIEVER_AVAILABLE = False
    print("⚠ retriever.py not found — Wikipedia evidence disabled")
    def get_evidence(claim, verbose=False):
        return {"evidence": "", "found": False}

# =============================================================
# CONFIGURATION
# =============================================================

CONFIG = {
    # Model paths — uses best checkpoints from each phase
    "detector_path"  : "./detector_model/best",
    "nli_path"       : "./nli_model/best",
    "corrector_path" : "./corrector_model/best",

    # Evaluation data
    "test_path"      : "./outputs/detection_test.csv",
    "benchmark_path" : "./outputs/benchmark_truthfulqa.csv",

    # Output folder for all results
    "output_dir"     : "./evaluation_results",

    # Model settings — must match training settings
    "max_length"         : 128,
    "max_input_length"   : 256,
    "max_output_length"  : 128,

    # How many TruthfulQA samples to evaluate
    # Set to None to evaluate all 790
    "benchmark_samples"  : 200,

    # Batch size for evaluation
    "batch_size"         : 16,

    "seed"               : 42,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# =============================================================
# STEP 1 — DEVICE SETUP
# =============================================================

print("=" * 60)
print("  Phase 7 — Full Pipeline Evaluation")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("\n⚠ Running on CPU")
print(f"   Device: {device}\n")

# =============================================================
# STEP 2 — LOAD ALL THREE MODELS
# =============================================================

print("--- Loading all trained models ---")

# ── Detection Model (Phase 3) ─────────────────────────────────
print("   Loading detection model...")
detector_tokenizer = AutoTokenizer.from_pretrained(CONFIG["detector_path"])
detector_model     = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["detector_path"]
).to(device)
detector_model.eval()
print("   ✅ Detection model loaded")

# ── NLI Model (Phase 4) ───────────────────────────────────────
print("   Loading NLI model...")
nli_tokenizer = AutoTokenizer.from_pretrained(CONFIG["nli_path"])
nli_model     = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["nli_path"]
).to(device)
nli_model.eval()
print("   ✅ NLI model loaded")

# ── Correction Model (Phase 6) ────────────────────────────────
print("   Loading correction model...")
corrector_tokenizer = T5Tokenizer.from_pretrained(
    CONFIG["corrector_path"], legacy=False
)
corrector_model = T5ForConditionalGeneration.from_pretrained(
    CONFIG["corrector_path"]
).to(device)
corrector_model.eval()
print("   ✅ Correction model loaded\n")

# =============================================================
# STEP 3 — INFERENCE FUNCTIONS
# One function per model for clean reusable code
# =============================================================

def detect_hallucination(statement, evidence=""):
    """
    Runs the Phase 3 detection model.
    Returns label, hallucination probability, and confidence.
    """
    encoding = detector_tokenizer(
        str(statement),
        str(evidence),
        max_length     = CONFIG["max_length"],
        truncation     = True,
        padding        = "max_length",
        return_tensors = "pt"
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = detector_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

    probs              = torch.softmax(outputs.logits, dim=1)[0]
    hallucination_prob = probs[1].item()

    return {
        "label"             : 1 if hallucination_prob > 0.5 else 0,
        "label_text"        : "Hallucinated" if hallucination_prob > 0.5 else "Factual",
        "hallucination_prob": round(hallucination_prob * 100, 2),
        "factual_prob"      : round(probs[0].item() * 100, 2),
    }


def verify_with_nli(premise, hypothesis):
    """
    Runs the Phase 4 NLI model.
    Returns relationship label and contradiction probability.
    In hallucination detection:
      premise    = evidence (what we know is true)
      hypothesis = statement being checked
    """
    encoding = nli_tokenizer(
        str(premise),
        str(hypothesis),
        max_length     = CONFIG["max_length"],
        truncation     = True,
        padding        = "max_length",
        return_tensors = "pt"
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = nli_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

    probs      = torch.softmax(outputs.logits, dim=1)[0]
    label_idx  = torch.argmax(probs).item()
    label_map  = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

    return {
        "label"             : label_map[label_idx],
        "entailment_prob"   : round(probs[0].item() * 100, 2),
        "neutral_prob"      : round(probs[1].item() * 100, 2),
        "contradiction_prob": round(probs[2].item() * 100, 2),
    }


def correct_statement(wrong_statement, evidence=""):
    """
    Runs the Phase 6 correction model.
    Generates a corrected version of the hallucinated statement.
    """
    if evidence and evidence.strip():
        input_text = f"correct: {wrong_statement} evidence: {evidence}"
    else:
        input_text = f"correct the following hallucinated statement: {wrong_statement}"

    inputs = corrector_tokenizer(
        input_text,
        max_length     = CONFIG["max_input_length"],
        truncation     = True,
        padding        = "max_length",
        return_tensors = "pt"
    )
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = corrector_model.generate(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            max_length           = CONFIG["max_output_length"],
            num_beams            = 4,
            early_stopping       = True,
            no_repeat_ngram_size = 2
        )

    corrected = corrector_tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    )
    return corrected


def run_full_pipeline(statement, evidence=""):
    """
    Runs all three models in sequence on one statement.
    This is the complete hallucination detection pipeline.

    Flow:
      1. Detection model → is it hallucinated?
      2. NLI model → does evidence contradict it?
      3. If hallucinated → correction model → fix it

    Returns a complete result dict.
    """
    # Step 1 — Detection
    detection = detect_hallucination(statement, evidence)

    # Step 2 — NLI verification (only if evidence available)
    nli_result = None
    if evidence and evidence.strip():
        nli_result = verify_with_nli(evidence, statement)

    # Step 3 — Correction (only if flagged as hallucinated)
    corrected = None
    if detection["label"] == 1:
        corrected = correct_statement(statement, evidence)

    # Combined verdict
    # If both detection AND NLI say hallucinated → HIGH confidence
    if nli_result:
        nli_contradiction = nli_result["contradiction_prob"] > 40
        if detection["label"] == 1 and nli_contradiction:
            confidence = "HIGH"
        elif detection["label"] == 1 or nli_contradiction:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
    else:
        confidence = "HIGH" if detection["hallucination_prob"] > 70 else "MEDIUM"

    return {
        "statement"         : statement,
        "evidence"          : evidence[:200] if evidence else "",
        "detection"         : detection,
        "nli"               : nli_result,
        "corrected"         : corrected,
        "final_label"       : detection["label"],
        "final_label_text"  : detection["label_text"],
        "pipeline_confidence": confidence,
    }


# =============================================================
# STEP 4 — EVALUATE DETECTION MODEL ON TEST SET
# =============================================================

print("=" * 60)
print("  PART 1: Detection Model Evaluation")
print("=" * 60)

df_test = pd.read_csv(CONFIG["test_path"], encoding="latin-1")
df_test = df_test.dropna(subset=["statement", "label"]).reset_index(drop=True)
df_test["evidence"]  = df_test["evidence"].fillna("").astype(str)
df_test["statement"] = df_test["statement"].astype(str)

print(f"\n   Test samples: {len(df_test)}")

# Run detection model on all test samples in batches
all_preds  = []
all_labels = []
all_probs  = []

print("   Running inference on test set...")

for i in range(0, len(df_test), CONFIG["batch_size"]):
    batch_df = df_test.iloc[i:i + CONFIG["batch_size"]]

    statements = batch_df["statement"].tolist()
    evidences  = batch_df["evidence"].tolist()
    labels     = batch_df["label"].tolist()

    for stmt, ev, lbl in zip(statements, evidences, labels):
        result = detect_hallucination(stmt, ev)
        all_preds.append(result["label"])
        all_labels.append(int(lbl))
        all_probs.append(result["hallucination_prob"] / 100)

    if (i // CONFIG["batch_size"] + 1) % 50 == 0:
        print(f"   Processed {i + CONFIG['batch_size']}/{len(df_test)}...")

# Classification metrics
accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="macro")
recall    = recall_score(all_labels, all_preds, average="macro")
f1        = f1_score(all_labels, all_preds, average="macro")
report    = classification_report(
    all_labels, all_preds,
    target_names=["Factual", "Hallucinated"]
)

print(f"\n   Detection Model Results:")
print(f"     Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"     Precision : {precision:.4f}")
print(f"     Recall    : {recall:.4f}")
print(f"     Macro F1  : {f1:.4f}")
print(f"\n{report}")

# ROC curve data
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc     = auc(fpr, tpr)
print(f"   ROC-AUC: {roc_auc:.4f}")

# =============================================================
# STEP 5 — EVALUATE ON TRUTHFULQA BENCHMARK
# =============================================================

print("\n" + "=" * 60)
print("  PART 2: TruthfulQA Benchmark Evaluation")
print("=" * 60)

df_bench = pd.read_csv(CONFIG["benchmark_path"], encoding="latin-1")

if CONFIG["benchmark_samples"]:
    df_bench = df_bench.sample(
        n            = min(CONFIG["benchmark_samples"], len(df_bench)),
        random_state = CONFIG["seed"]
    ).reset_index(drop=True)

print(f"\n   Benchmark samples : {len(df_bench)}")
print(f"   Categories        : {df_bench['category'].nunique()}")
print("   Running full pipeline on benchmark...\n")

benchmark_results = []
correct_count     = 0
corrected_texts   = []
reference_texts   = []

for i, row in df_bench.iterrows():
    question       = str(row["question"])
    correct_ans    = str(row["correct_answer"])
    incorrect_ans  = str(row["incorrect_answer"])
    category       = str(row["category"])

    # Get Wikipedia evidence if retriever available
    if RETRIEVER_AVAILABLE:
        ev_result = get_evidence(question, verbose=False)
        evidence  = ev_result["evidence"]
    else:
        evidence = ""

    # Run detection on correct answer (should be Factual = 0)
    correct_result   = detect_hallucination(correct_ans, evidence)
    # Run detection on incorrect answer (should be Hallucinated = 1)
    incorrect_result = detect_hallucination(incorrect_ans, evidence)

    # Model is correct if:
    #   correct answer   classified as Factual (0)
    #   incorrect answer classified as Hallucinated (1)
    model_correct = (
        correct_result["label"] == 0 and
        incorrect_result["label"] == 1
    )
    if model_correct:
        correct_count += 1

    # Run correction on incorrect answer
    corrected = correct_statement(incorrect_ans, evidence)
    corrected_texts.append(corrected)
    reference_texts.append(correct_ans)

    benchmark_results.append({
        "question"          : question,
        "correct_answer"    : correct_ans,
        "incorrect_answer"  : incorrect_ans,
        "category"          : category,
        "correct_pred"      : correct_result["label"],
        "incorrect_pred"    : incorrect_result["label"],
        "model_correct"     : model_correct,
        "corrected_output"  : corrected,
        "evidence_found"    : bool(evidence),
    })

    if (i + 1) % 20 == 0:
        print(f"   Processed {i+1}/{len(df_bench)}...")

df_results = pd.DataFrame(benchmark_results)
benchmark_accuracy = correct_count / len(df_bench)

print(f"\n   TruthfulQA Benchmark Accuracy: {benchmark_accuracy:.4f} ({benchmark_accuracy*100:.2f}%)")
print(f"   Correct predictions: {correct_count}/{len(df_bench)}")

# Per-category accuracy
print(f"\n   Per-category accuracy:")
category_acc = df_results.groupby("category")["model_correct"].mean().sort_values(ascending=False)
for cat, acc in category_acc.head(10).items():
    print(f"     {cat:<35}: {acc:.2f}")

# =============================================================
# STEP 6 — ROUGE SCORES FOR CORRECTION ENGINE
# =============================================================

print("\n" + "=" * 60)
print("  PART 3: Correction Engine ROUGE Evaluation")
print("=" * 60)

rouge_scores = {}
if ROUGE_AVAILABLE and corrected_texts:
    scores = rouge.compute(
        predictions = corrected_texts,
        references  = reference_texts
    )
    rouge_scores = {
        "ROUGE-1": round(scores["rouge1"] * 100, 2),
        "ROUGE-2": round(scores["rouge2"] * 100, 2),
        "ROUGE-L": round(scores["rougeL"] * 100, 2),
    }
    print(f"\n   Correction Engine ROUGE Scores:")
    for k, v in rouge_scores.items():
        print(f"     {k} : {v}")
    print(f"\n   Interpretation:")
    print(f"     ROUGE-1 measures individual word overlap")
    print(f"     ROUGE-2 measures two-word phrase overlap")
    print(f"     ROUGE-L measures longest matching sequence")
    print(f"     Score > 30 is considered good for correction tasks")
else:
    print("   ROUGE scoring skipped (evaluate library not available)")

# =============================================================
# STEP 7 — SAVE ALL RESULTS
# =============================================================

print("\n" + "=" * 60)
print("  Saving Results")
print("=" * 60)

# Save benchmark results CSV
results_csv_path = os.path.join(CONFIG["output_dir"], "pipeline_results.csv")
df_results.to_csv(results_csv_path, index=False)
print(f"\n✅ Pipeline results saved: {results_csv_path}")

# Save full text report
report_path = os.path.join(CONFIG["output_dir"], "full_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("  HALLUCINATION DETECTION ENGINE — FULL EVALUATION REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write("DETECTION MODEL (Phase 3 — RoBERTa)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Test Samples : {len(df_test)}\n")
    f.write(f"Accuracy     : {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision    : {precision:.4f}\n")
    f.write(f"Recall       : {recall:.4f}\n")
    f.write(f"Macro F1     : {f1:.4f}\n")
    f.write(f"ROC-AUC      : {roc_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n\n")

    f.write("TRUTHFULQA BENCHMARK (Full Pipeline)\n")
    f.write("-" * 40 + "\n")
    f.write(f"Samples Evaluated : {len(df_bench)}\n")
    f.write(f"Pipeline Accuracy : {benchmark_accuracy:.4f} ({benchmark_accuracy*100:.2f}%)\n\n")
    f.write("Per-Category Accuracy:\n")
    for cat, acc in category_acc.items():
        f.write(f"  {cat:<35}: {acc:.2f}\n")
    f.write("\n")

    if rouge_scores:
        f.write("CORRECTION ENGINE ROUGE SCORES (Phase 6 — Flan-T5)\n")
        f.write("-" * 40 + "\n")
        for k, v in rouge_scores.items():
            f.write(f"  {k} : {v}\n")

print(f"✅ Full report saved: {report_path}")

# =============================================================
# STEP 8 — GENERATE ALL PLOTS
# =============================================================

print("\n--- Generating evaluation plots ---")

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── Plot 1: Confusion Matrix ──────────────────────────────────
ax1  = fig.add_subplot(gs[0, 0])
cm   = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Factual", "Hallucinated"])
disp.plot(ax=ax1, colorbar=False, cmap="Blues")
ax1.set_title("Detection Model\nConfusion Matrix", fontweight="bold", fontsize=12)

# ── Plot 2: ROC Curve ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(fpr, tpr, color="darkorange", lw=2,
         label=f"ROC curve (AUC = {roc_auc:.3f})")
ax2.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve\n(Detection Model)", fontweight="bold", fontsize=12)
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Plot 3: Metrics Bar Chart ─────────────────────────────────
ax3     = fig.add_subplot(gs[0, 2])
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
values  = [accuracy, precision, recall, f1, roc_auc]
colors  = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
bars    = ax3.bar(metrics, values, color=colors, alpha=0.85, edgecolor="white")
ax3.set_ylim(0, 1.15)
ax3.set_ylabel("Score")
ax3.set_title("Detection Model\nMetrics Summary", fontweight="bold", fontsize=12)
ax3.grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# ── Plot 4: Per-Category TruthfulQA Accuracy ─────────────────
ax4      = fig.add_subplot(gs[1, :2])
top_cats = category_acc.head(15)
colors_cat = ["#4CAF50" if v >= 0.6 else "#FF9800" if v >= 0.4 else "#F44336"
              for v in top_cats.values]
bars4 = ax4.barh(top_cats.index, top_cats.values, color=colors_cat, alpha=0.85)
ax4.set_xlabel("Accuracy")
ax4.set_title("TruthfulQA Per-Category Pipeline Accuracy\n(Top 15 Categories)",
              fontweight="bold", fontsize=12)
ax4.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="50% baseline")
ax4.set_xlim(0, 1.1)
ax4.legend(fontsize=9)
ax4.grid(True, axis="x", alpha=0.3)
for bar, val in zip(bars4, top_cats.values):
    ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", fontsize=8)

# ── Plot 5: ROUGE Scores ──────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
if rouge_scores:
    rouge_names  = list(rouge_scores.keys())
    rouge_values = [v/100 for v in rouge_scores.values()]
    bars5 = ax5.bar(rouge_names, rouge_values,
                    color=["#3F51B5", "#009688", "#FF5722"],
                    alpha=0.85, edgecolor="white")
    ax5.set_ylim(0, 1.0)
    ax5.axhline(y=0.3, color="green", linestyle="--", alpha=0.6, label="Good threshold (0.30)")
    ax5.set_ylabel("Score")
    ax5.set_title("Correction Engine\nROUGE Scores", fontweight="bold", fontsize=12)
    ax5.legend(fontsize=9)
    ax5.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars5, rouge_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
else:
    ax5.text(0.5, 0.5, "ROUGE scores\nnot available",
             ha="center", va="center", transform=ax5.transAxes, fontsize=12)
    ax5.set_title("Correction Engine\nROUGE Scores", fontweight="bold", fontsize=12)

fig.suptitle("Hallucination Detection & Correction Engine — Full Evaluation",
             fontsize=16, fontweight="bold", y=1.01)

plot_path = os.path.join(CONFIG["output_dir"], "full_evaluation.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"✅ Evaluation plots saved: {plot_path}")

# =============================================================
# STEP 9 — FULL PIPELINE DEMO
# Show end-to-end examples
# =============================================================

print("\n" + "=" * 60)
print("  PART 4: Full Pipeline Demo")
print("=" * 60)

demo_cases = [
    {
        "statement": "Albert Einstein invented the telephone in 1876.",
        "evidence" : "Alexander Graham Bell is credited with patenting the first practical telephone in 1876."
    },
    {
        "statement": "The Eiffel Tower is located in Berlin, Germany.",
        "evidence" : "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
    },
    {
        "statement": "Water boils at 100 degrees Celsius at sea level.",
        "evidence" : "Water boils at 100 degrees Celsius or 212 degrees Fahrenheit at standard atmospheric pressure."
    },
    {
        "statement": "The Amazon is the longest river in the world.",
        "evidence" : "The Nile is generally considered the longest river at 6,650 km. The Amazon is largest by flow."
    },
    {
        "statement": "Shakespeare was born in London.",
        "evidence" : "William Shakespeare was born in Stratford-upon-Avon, Warwickshire, England in April 1564."
    },
]

print()
for i, case in enumerate(demo_cases, 1):
    result = run_full_pipeline(case["statement"], case["evidence"])

    det    = result["detection"]
    nli    = result["nli"]
    flag   = "⚠" if result["final_label"] == 1 else "✅"

    print(f"  {'─'*56}")
    print(f"  Example {i}: {flag} {result['final_label_text'].upper()}")
    print(f"  Statement  : {case['statement']}")
    print(f"  Detection  : {det['hallucination_prob']}% hallucination probability")

    if nli:
        print(f"  NLI Result : {nli['label']} "
              f"(Contradiction: {nli['contradiction_prob']}%)")

    print(f"  Confidence : {result['pipeline_confidence']}")

    if result["corrected"]:
        print(f"  Corrected  : {result['corrected']}")
    print()

# =============================================================
# STEP 10 — PRINT FINAL SUMMARY
# =============================================================

print("=" * 60)
print("  EVALUATION COMPLETE — SUMMARY")
print("=" * 60)
print(f"\n  Detection Model (RoBERTa):")
print(f"    Accuracy  : {accuracy*100:.2f}%")
print(f"    Macro F1  : {f1:.4f}")
print(f"    ROC-AUC   : {roc_auc:.4f}")
print(f"\n  Full Pipeline (TruthfulQA Benchmark):")
print(f"    Accuracy  : {benchmark_accuracy*100:.2f}%")
if rouge_scores:
    print(f"\n  Correction Engine (Flan-T5):")
    for k, v in rouge_scores.items():
        print(f"    {k}      : {v}")
print(f"\n  Output files saved to: {os.path.abspath(CONFIG['output_dir'])}/")
print(f"    full_report.txt")
print(f"    full_evaluation.png")
print(f"    pipeline_results.csv")
print("\n  Use these results in your MSc project report.")
print("=" * 60)
