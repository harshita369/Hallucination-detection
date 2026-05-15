"""
=============================================================
  Hallucination Detection Engine — Unified Preprocessing
=============================================================
Files used (ALL CSV):
  1. shared_task_dev.csv            → FEVER Dev      (19,998 rows)
  2. train.csv                      → FEVER Train    (145,449 rows)
  3. qa_data.csv                    → HaluEval QA    (10,000 rows)
  4. summarization.csv              → HaluEval Summ  (10,001 rows)
  5. general_data.csv               → General Data   (4,507 rows)
  6. TruthfulQA.csv                 → Benchmark      (790 rows)
  7. train-00000-of-00001.csv       → SNLI Train     (~550,000 rows)
  8. validation-00000-of-00001.csv  → SNLI Val       (10,000 rows)
  9. test-00000-of-00001.csv        → SNLI Test      (10,000 rows)

Outputs saved to ./outputs/ folder:
  → detection_train.csv
  → detection_test.csv
  → correction_train.csv
  → nli_train.csv
  → nli_val.csv
  → nli_test.csv
  → benchmark_truthfulqa.csv
  → dataset_summary.txt
=============================================================
"""

import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ── CONFIG ─────────────────────────────────────────────────
DATA_DIR   = "./"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

log("=============================================================")
log("  Hallucination Detection Engine — Preprocessing Started")
log("=============================================================")


# =============================================================
# STEP 1: FEVER (shared_task_dev.csv + train.csv)
# =============================================================
log("\n--- STEP 1: Loading FEVER datasets ---")

def process_fever_csv(path, name):
    df = pd.read_csv(path, encoding="latin-1")
    df = df[df["label"] != "NOT ENOUGH INFO"].copy()

    def extract_title(ev):
        try:
            matches = re.findall(r"'([^']+)'", str(ev))
            if matches:
                return matches[0].replace("_", " ").replace("-LRB-", "(").replace("-RRB-", ")")
        except:
            pass
        return ""

    records = []
    for _, row in df.iterrows():
        records.append({
            "statement":      str(row["claim"]),
            "evidence":       extract_title(row["evidence"]),
            "label":          1 if row["label"] == "REFUTES" else 0,
            "source_dataset": name
        })
    return pd.DataFrame(records)

df_fever_dev   = process_fever_csv(os.path.join(DATA_DIR, "shared_task_dev.csv"), "FEVER_dev")
df_fever_train = process_fever_csv(os.path.join(DATA_DIR, "train.csv"),           "FEVER_train")
df_fever       = pd.concat([df_fever_dev, df_fever_train], ignore_index=True)

log(f"FEVER Dev loaded:   {len(df_fever_dev)} rows")
log(f"FEVER Train loaded: {len(df_fever_train)} rows")
log(f"FEVER Total:        {len(df_fever)} rows")
log(f"  Factual (0): {(df_fever['label']==0).sum()} | Hallucinated (1): {(df_fever['label']==1).sum()}")


# =============================================================
# STEP 2: HaluEval QA (qa_data.csv)
# =============================================================
log("\n--- STEP 2: Loading HaluEval QA ---")

df_qa = pd.read_csv(os.path.join(DATA_DIR, "qa_data.csv"), encoding="latin-1")

qa_detection  = []
qa_correction = []

for _, row in df_qa.iterrows():
    evidence   = str(row["knowledge"])
    question   = str(row["question"])
    right_ans  = str(row["right_answer"])
    halluc_ans = str(row["hallucinated_answer"])

    qa_detection.append({"statement": right_ans,  "evidence": evidence, "label": 0, "source_dataset": "HaluEval_QA"})
    qa_detection.append({"statement": halluc_ans, "evidence": evidence, "label": 1, "source_dataset": "HaluEval_QA"})

    qa_correction.append({
        "wrong_statement":   halluc_ans,
        "evidence":          evidence,
        "correct_statement": right_ans,
        "question":          question,
        "source_dataset":    "HaluEval_QA"
    })

df_qa_det  = pd.DataFrame(qa_detection)
df_qa_corr = pd.DataFrame(qa_correction)

log(f"HaluEval QA detection rows:  {len(df_qa_det)} (perfectly balanced)")
log(f"HaluEval QA correction rows: {len(df_qa_corr)}")


# =============================================================
# STEP 3: HaluEval Summarization (summarization.csv)
# =============================================================
log("\n--- STEP 3: Loading HaluEval Summarization ---")

df_summ = pd.read_csv(os.path.join(DATA_DIR, "summarization.csv"), encoding="latin-1")

summ_detection  = []
summ_correction = []

for _, row in df_summ.iterrows():
    document    = str(row["document"])[:500]
    right_summ  = str(row["right_summary"])
    halluc_summ = str(row["hallucinated_summary"])

    summ_detection.append({"statement": right_summ,  "evidence": document, "label": 0, "source_dataset": "HaluEval_Summarization"})
    summ_detection.append({"statement": halluc_summ, "evidence": document, "label": 1, "source_dataset": "HaluEval_Summarization"})

    summ_correction.append({
        "wrong_statement":   halluc_summ,
        "evidence":          document,
        "correct_statement": right_summ,
        "question":          "Summarize the document correctly.",
        "source_dataset":    "HaluEval_Summarization"
    })

df_summ_det  = pd.DataFrame(summ_detection)
df_summ_corr = pd.DataFrame(summ_correction)

log(f"Summarization detection rows:  {len(df_summ_det)} (perfectly balanced)")
log(f"Summarization correction rows: {len(df_summ_corr)}")


# =============================================================
# STEP 4: General Hallucination Data (general_data.csv)
# =============================================================
log("\n--- STEP 4: Loading General Hallucination Data ---")

df_gen = pd.read_csv(os.path.join(DATA_DIR, "general_data.csv"), encoding="latin-1")
df_gen["label_num"] = df_gen["hallucination"].map({"yes": 1, "no": 0})
df_gen = df_gen.dropna(subset=["label_num"])

log(f"Before balancing — Factual: {(df_gen['label_num']==0).sum()} | Hallucinated: {(df_gen['label_num']==1).sum()}")

majority    = df_gen[df_gen["label_num"] == 0]
minority    = df_gen[df_gen["label_num"] == 1]
minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
df_gen_bal  = pd.concat([majority, minority_up]).sample(frac=1, random_state=42)

df_gen_det = pd.DataFrame({
    "statement":      df_gen_bal["chatgpt_response"].astype(str),
    "evidence":       df_gen_bal["user_query"].astype(str),
    "label":          df_gen_bal["label_num"].astype(int),
    "source_dataset": "GeneralHallucination"
})

log(f"After balancing: {len(df_gen_det)} rows (each class: {len(majority)})")


# =============================================================
# STEP 5: TruthfulQA — benchmark only, NOT used in training
# =============================================================
log("\n--- STEP 5: Loading TruthfulQA (benchmark only) ---")

df_tqa = pd.read_csv(os.path.join(DATA_DIR, "TruthfulQA.csv"), encoding="latin-1")

df_benchmark = pd.DataFrame({
    "question":         df_tqa["Question"].astype(str),
    "correct_answer":   df_tqa["Best Answer"].astype(str),
    "incorrect_answer": df_tqa["Best Incorrect Answer"].astype(str),
    "category":         df_tqa["Category"].astype(str),
    "source_dataset":   "TruthfulQA"
})

log(f"TruthfulQA: {len(df_benchmark)} questions | {df_tqa['Category'].nunique()} categories")
log("  ⚠ Reserved for FINAL EVALUATION only — not used in training")


# =============================================================
# STEP 6: SNLI (train / validation / test)
# =============================================================
log("\n--- STEP 6: Loading SNLI datasets ---")

def load_snli(path, split_name):
    df = pd.read_csv(path, encoding="latin-1")
    df = df[df["label"] != -1].copy()
    df = df.rename(columns={"label": "nli_label"})
    df["split"] = split_name
    return df[["premise", "hypothesis", "nli_label", "split"]]

df_snli_train = load_snli(os.path.join(DATA_DIR, "train-00000-of-00001.csv"),      "train")
df_snli_val   = load_snli(os.path.join(DATA_DIR, "validation-00000-of-00001.csv"), "validation")
df_snli_test  = load_snli(os.path.join(DATA_DIR, "test-00000-of-00001.csv"),       "test")

log(f"SNLI Train: {len(df_snli_train)} | Val: {len(df_snli_val)} | Test: {len(df_snli_test)}")
log("  Labels → 0: Entailment  |  1: Neutral  |  2: Contradiction")


# =============================================================
# STEP 7: Merge all detection data + 80/20 train/test split
# =============================================================
log("\n--- STEP 7: Merging all detection datasets ---")

df_detection = pd.concat([
    df_fever,
    df_qa_det,
    df_summ_det,
    df_gen_det
], ignore_index=True)

df_detection["statement"] = df_detection["statement"].fillna("").str.strip()
df_detection["evidence"]  = df_detection["evidence"].fillna("").str.strip()
df_detection["label"]     = df_detection["label"].astype(int)
df_detection = df_detection[df_detection["statement"].str.len() > 5].reset_index(drop=True)

log(f"Total detection samples: {len(df_detection)}")
log(f"  Factual (0):      {(df_detection['label']==0).sum()}")
log(f"  Hallucinated (1): {(df_detection['label']==1).sum()}")
log(f"  Breakdown by source:")
for src, grp in df_detection.groupby("source_dataset"):
    log(f"    {src}: {len(grp)} rows")

df_det_train, df_det_test = train_test_split(
    df_detection, test_size=0.2, random_state=42, stratify=df_detection["label"]
)
log(f"\nDetection Train: {len(df_det_train)} | Test: {len(df_det_test)}")


# =============================================================
# STEP 8: Merge all correction data
# =============================================================
log("\n--- STEP 8: Merging correction datasets ---")

df_correction = pd.concat([df_qa_corr, df_summ_corr], ignore_index=True)
df_correction["wrong_statement"]   = df_correction["wrong_statement"].fillna("").str.strip()
df_correction["correct_statement"] = df_correction["correct_statement"].fillna("").str.strip()
df_correction["evidence"]          = df_correction["evidence"].fillna("").str.strip()

log(f"Total correction samples: {len(df_correction)}")


# =============================================================
# STEP 9: Save all output files
# =============================================================
log("\n--- STEP 9: Saving all output files ---")

df_det_train.to_csv(  os.path.join(OUTPUT_DIR, "detection_train.csv"),      index=False)
df_det_test.to_csv(   os.path.join(OUTPUT_DIR, "detection_test.csv"),       index=False)
df_correction.to_csv( os.path.join(OUTPUT_DIR, "correction_train.csv"),     index=False)
df_snli_train.to_csv( os.path.join(OUTPUT_DIR, "nli_train.csv"),            index=False)
df_snli_val.to_csv(   os.path.join(OUTPUT_DIR, "nli_val.csv"),              index=False)
df_snli_test.to_csv(  os.path.join(OUTPUT_DIR, "nli_test.csv"),             index=False)
df_benchmark.to_csv(  os.path.join(OUTPUT_DIR, "benchmark_truthfulqa.csv"), index=False)

log(f"✅ detection_train.csv      ({len(df_det_train)} rows)")
log(f"✅ detection_test.csv       ({len(df_det_test)} rows)")
log(f"✅ correction_train.csv     ({len(df_correction)} rows)")
log(f"✅ nli_train.csv            ({len(df_snli_train)} rows)")
log(f"✅ nli_val.csv              ({len(df_snli_val)} rows)")
log(f"✅ nli_test.csv             ({len(df_snli_test)} rows)")
log(f"✅ benchmark_truthfulqa.csv ({len(df_benchmark)} rows)")

with open(os.path.join(OUTPUT_DIR, "dataset_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"✅ dataset_summary.txt")

log("\n=============================================================")
log("  ✅ Preprocessing Complete!")
log(f"  All files saved to: {os.path.abspath(OUTPUT_DIR)}")
log("=============================================================")


# =============================================================
# STEP 10: Sanity check
# =============================================================
print("\n\n--- SANITY CHECK ---")
print("\nDetection Train sample:")
print(df_det_train[["statement", "label", "source_dataset"]].head(3).to_string())
print("\nCorrection Train sample:")
print(df_correction[["wrong_statement", "correct_statement"]].head(2).to_string())
print("\nNLI Train sample:")
print(df_snli_train[["premise", "hypothesis", "nli_label"]].head(2).to_string())
print("\nBenchmark sample:")
print(df_benchmark[["question", "correct_answer", "category"]].head(2).to_string())
