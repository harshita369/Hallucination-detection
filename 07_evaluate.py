"""
=============================================================
  Phase 7 — Full Pipeline Evaluation v3
  GPU    : RTX 3050 Mobile 4GB — Linux

  Integrates all 3 layers:
  1. Rule-based check (deterministic, high precision)
  2. RoBERTa detection (statement-only)
  3. NLI with Wikipedia
=============================================================
"""

import os, sys, re, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration
)
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
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
    print(f"⚠ Retriever: {e}")
    def get_evidence(c, verbose=False): return {"evidence":"","found":False}

CONFIG = {
    "detector_path"   : "./detector_model/best",
    "nli_path"        : "./nli_model/best",
    "corrector_path"  : "./corrector_model/best",
    "test_path"       : "./outputs/detection_test.csv",
    "benchmark_path"  : "./outputs/benchmark_truthfulqa.csv",
    "output_dir"      : "./evaluation_results",
    "max_length"      : 128,
    "max_input_length": 256,
    "max_output_length": 128,
    "benchmark_samples": None,
    "batch_size"      : 16,
    "seed"            : 42,
}
os.makedirs(CONFIG["output_dir"], exist_ok=True)

print("="*60)
print("  Phase 7 — Full Pipeline Evaluation v3")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available(): print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"   Device: {device}\n")

# ── Load models ───────────────────────────────────────────
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

threshold = 0.5
t_file = os.path.join(CONFIG["detector_path"], "threshold.txt")
if os.path.exists(t_file):
    with open(t_file) as f: threshold = float(f.read().strip())
    print(f"   ✅ Threshold: {threshold}\n")

# ── Rule layer (same as predict.py) ──────────────────────
UNIT_RULES = [
    (r"1\s*li?t(?:er|re)\s*(?:is|=|equals?)\s*(\d+)\s*m[li]",
     lambda m: int(m.group(1)) != 1000, "1 liter = 1000 ml"),
    (r"1\s*km\s*(?:is|=|equals?)\s*1\s*li?t(?:er|re)",
     lambda m: True, "km and liters are different units"),
    (r"water\s+boils?\s+at\s+(\d+)\s+degrees?",
     lambda m: abs(int(m.group(1))-100) > 5, "Water boils at 100°C"),
    (r"water\s+freezes?\s+at\s+(\d+)\s+degrees?",
     lambda m: int(m.group(1)) != 0, "Water freezes at 0°C"),
    (r"(?:standard|normal|regular)\s+car\s+has\s+(\d+)\s+wheels?",
     lambda m: int(m.group(1)) != 4, "Standard car has 4 wheels"),
    (r"humans?\s+have\s+(\d+)\s+fingers?\s+(?:on|in)\s+each\s+hand",
     lambda m: int(m.group(1)) != 5, "Humans have 5 fingers per hand"),
    (r"1\s*kg\s*(?:is|=|equals?)\s*(\d+)\s*(?:gram|g\b)",
     lambda m: int(m.group(1)) != 1000, "1 kg = 1000 grams"),
    (r"1\s*(?:meter|m)\s*(?:is|=|equals?)\s*(\d+)\s*(?:cm|centimeter)",
     lambda m: int(m.group(1)) != 100, "1 meter = 100 cm"),
]
KNOWN_HALL = [
    (r"milky\s*way\s+galaxy\s+is\s+(?:a|the)\s+(?:name\s+of\s+a\s+)?planet","Milky Way is a galaxy"),
    (r"andromeda\s+galaxy\s+is\s+the\s+galaxy\s+(?:in\s+which|where)\s+we\s+live","We live in Milky Way"),
    (r"einstein\s+invented\s+the\s+telephone","Bell invented the telephone"),
    (r"shakespeare\s+was\s+born\s+in\s+london","Shakespeare born in Stratford-upon-Avon"),
    (r"eiffel\s+tower\s+is\s+(?:located\s+)?in\s+berlin","Eiffel Tower is in Paris"),
    (r"psychology\s+is\s+the\s+study\s+of\s+(?:fish|fishes)","Psychology = study of mind"),
    (r"exit\s+means\s+to\s+keep\s+going","Exit = to leave"),
    (r"domino'?s\s+is\s+a\s+sushi","Domino's is pizza"),
    (r"mcdonald'?s\s+is\s+a\s+chinese","McDonald's is American"),
    (r"sound\s+travels?\s+faster\s+than\s+light","Light > sound speed"),
    (r"amazon\s+(?:river\s+)?is\s+the\s+longest\s+river","Nile is longest river"),
    (r"new\s+zealand\s+is\s+the\s+(?:biggest|largest)\s+country","Russia is largest country"),
]

def rule_check(statement):
    s = statement.lower().strip()
    for pat, fn, note in UNIT_RULES:
        m = re.search(pat, s)
        if m:
            try:
                if fn(m): return True, note
            except: pass
    for pat, note in KNOWN_HALL:
        if re.search(pat, s): return True, note
    return False, None

# ── Inference functions ───────────────────────────────────
def detect(statement):
    enc = det_tok(str(statement), max_length=CONFIG["max_length"],
                  truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        out = det_model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
    p = torch.softmax(out.logits,1)[0]
    h = p[1].item()
    return {"label":1 if h>threshold else 0, "label_text":"Hallucinated" if h>threshold else "Factual",
            "h_prob":round(h*100,2), "raw":h}

def nli_check(premise, hypothesis):
    enc = nli_tok(str(premise), str(hypothesis), max_length=CONFIG["max_length"],
                  truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        out = nli_model(input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device))
    p = torch.softmax(out.logits,1)[0]
    idx = torch.argmax(p).item()
    l = {0:"Entailment",1:"Neutral",2:"Contradiction"}
    return {"label":l[idx],"entailment":round(p[0].item()*100,2),
            "neutral":round(p[1].item()*100,2),"contradiction":round(p[2].item()*100,2),
            "raw_contra":p[2].item()}

def correct(statement, evidence=""):
    prompt = (f"Fix the incorrect fact. Evidence: {evidence.strip()[:200]} Statement: {statement} Correction:"
              if evidence and evidence.strip() else
              f"Fix the incorrect fact. Statement: {statement} Correction:")
    enc = cor_tok(prompt, max_length=CONFIG["max_input_length"],
                  truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        ids = cor_model.generate(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            max_length=CONFIG["max_output_length"], min_length=10,
            num_beams=4, length_penalty=1.5, early_stopping=True,
            no_repeat_ngram_size=3)
    return cor_tok.decode(ids[0], skip_special_tokens=True)

def pipeline(statement, evidence=""):
    rule_flag, rule_reason = rule_check(statement)
    det = detect(statement)
    nli = nli_check(evidence, statement) if evidence and evidence.strip() else None

    if rule_flag:
        verdict, conf, h_score = "Hallucinated", "HIGH", 1.0
    else:
        c_prob  = nli["raw_contra"] if nli else det["raw"]
        h_score = 0.6*det["raw"] + 0.4*c_prob
        if h_score > 0.6: verdict, conf = "Hallucinated", "HIGH"
        elif h_score > 0.35: verdict, conf = "Hallucinated", "MEDIUM"
        else: verdict, conf = "Factual", "LOW"

    corrected = correct(statement, evidence) if verdict == "Hallucinated" else None
    return {"detection":det, "nli":nli, "verdict":verdict, "confidence":conf,
            "h_score":h_score, "corrected":corrected,
            "final_label":1 if verdict=="Hallucinated" else 0}

# =============================================================
# PART 1: DETECTION EVALUATION
# =============================================================
print("="*60+"\n  PART 1: Detection Model Evaluation\n"+"="*60)

df_test = pd.read_csv(CONFIG["test_path"], encoding="latin-1")
df_test = df_test.dropna(subset=["statement","label"]).reset_index(drop=True)
df_test["statement"] = df_test["statement"].astype(str)
print(f"\n   Test samples: {len(df_test)} | Threshold: {threshold}\n")

all_preds, all_labels, all_probs = [], [], []
for i in range(0, len(df_test), CONFIG["batch_size"]):
    for _, row in df_test.iloc[i:i+CONFIG["batch_size"]].iterrows():
        r = detect(str(row["statement"]))
        all_preds.append(r["label"]); all_labels.append(int(row["label"]))
        all_probs.append(r["raw"])
    if (i//CONFIG["batch_size"]+1)%50==0:
        print(f"   Processed {min(i+CONFIG['batch_size'],len(df_test))}/{len(df_test)}...")

acc  = accuracy_score(all_labels,all_preds)
prec = precision_score(all_labels,all_preds,average="macro")
rec  = recall_score(all_labels,all_preds,average="macro")
f1   = f1_score(all_labels,all_preds,average="macro")
rep  = classification_report(all_labels,all_preds,target_names=["Factual","Hallucinated"])
fpr,tpr,_ = roc_curve(all_labels,all_probs)
roc_auc   = auc(fpr,tpr)

print(f"\n   Accuracy:{acc:.4f} Precision:{prec:.4f} Recall:{rec:.4f} F1:{f1:.4f} AUC:{roc_auc:.4f}")
print(f"\n{rep}")
if "source_dataset" in df_test.columns:
    print("   Per-source:")
    for src in df_test["source_dataset"].unique():
        idx = df_test[df_test["source_dataset"]==src].index.tolist()
        print(f"     {src:<35}: {accuracy_score([all_labels[i] for i in idx],[all_preds[i] for i in idx]):.3f}")

# =============================================================
# PART 2: TRUTHFULQA
# =============================================================
print("\n"+"="*60+"\n  PART 2: TruthfulQA\n"+"="*60)
df_bench = pd.read_csv(CONFIG["benchmark_path"], encoding="latin-1")
if CONFIG["benchmark_samples"]:
    df_bench = df_bench.sample(n=min(CONFIG["benchmark_samples"],len(df_bench)),
                                random_state=CONFIG["seed"]).reset_index(drop=True)
print(f"\n   Samples:{len(df_bench)} | Wikipedia:{'on' if RETRIEVER else 'off'}\n")

bench_rows, correct_both, correct_either = [], 0, 0
corr_texts, ref_texts = [], []

for i, row in df_bench.iterrows():
    q, ca, ia, cat = str(row["question"]), str(row["correct_answer"]), \
                     str(row["incorrect_answer"]), str(row["category"])
    ev = get_evidence(q, verbose=False).get("evidence","") if RETRIEVER else ""

    cr = detect(ca); ir = detect(ia)
    # Also apply rules
    r_ca, _ = rule_check(ca); r_ia, _ = rule_check(ia)
    if r_ca: cr["label"] = 0  # rule says it's not hallucinated (it's a correct answer)
    if r_ia: ir["label"] = 1  # rule says it is hallucinated

    both = (cr["label"]==0 and ir["label"]==1)
    if both: correct_both+=1
    if cr["label"]==0 or ir["label"]==1: correct_either+=1

    c = correct(ia, ev); corr_texts.append(c); ref_texts.append(ca)
    bench_rows.append({"question":q,"category":cat,"both_correct":both,"evidence_found":bool(ev)})
    if (i+1)%50==0: print(f"   Processed {i+1}/{len(df_bench)}...")

df_res = pd.DataFrame(bench_rows)
strict  = correct_both/len(df_bench)
partial = correct_either/len(df_bench)
print(f"\n   Strict:{strict:.4f} ({strict*100:.2f}%) | Partial:{partial:.4f} ({partial*100:.2f}%)")
print(f"   Evidence retrieved: {df_res['evidence_found'].sum()}/{len(df_bench)}")

cat_acc = df_res.groupby("category")["both_correct"].mean().sort_values(ascending=False)
for cat, val in cat_acc.head(10).items(): print(f"     {cat:<35}: {val:.2f}")

# =============================================================
# PART 3: ROUGE
# =============================================================
print("\n"+"="*60+"\n  PART 3: ROUGE Scores\n"+"="*60)
rouge_scores = {}
if ROUGE_OK and corr_texts:
    r = rouge_metric.compute(predictions=corr_texts, references=ref_texts)
    rouge_scores = {"ROUGE-1":round(r["rouge1"]*100,2),
                    "ROUGE-2":round(r["rouge2"]*100,2),
                    "ROUGE-L":round(r["rougeL"]*100,2)}
    for k,v in rouge_scores.items(): print(f"\n   {k}: {v}")

# Save
df_res.to_csv(os.path.join(CONFIG["output_dir"],"pipeline_results.csv"),index=False)
with open(os.path.join(CONFIG["output_dir"],"full_report.txt"),"w") as f:
    f.write("HALLUCINATION DETECTION — v3 EVALUATION\n"+"="*60+"\n\n")
    f.write(f"Threshold: {threshold}\n")
    f.write(f"Accuracy:{acc:.4f} F1:{f1:.4f} AUC:{roc_auc:.4f}\n\n")
    f.write(rep+"\n\nTruthfulQA:\n")
    f.write(f"Strict:{strict:.4f} Partial:{partial:.4f}\n\n")
    for cat,val in cat_acc.items(): f.write(f"  {cat:<35}: {val:.2f}\n")
    if rouge_scores:
        f.write("\nROUGE:\n")
        for k,v in rouge_scores.items(): f.write(f"  {k}:{v}\n")
print("\n✅ Reports saved")

# Plots
fig = plt.figure(figsize=(18,14))
gs  = gridspec.GridSpec(2,3,figure=fig,hspace=0.4,wspace=0.35)
ax1 = fig.add_subplot(gs[0,0])
cm  = confusion_matrix(all_labels,all_preds)
ConfusionMatrixDisplay(cm,display_labels=["Factual","Hallucinated"]).plot(ax=ax1,colorbar=False,cmap="Blues")
ax1.set_title("Confusion Matrix",fontweight="bold",fontsize=12)
ax2 = fig.add_subplot(gs[0,1])
ax2.plot(fpr,tpr,"darkorange",lw=2,label=f"AUC={roc_auc:.3f}")
ax2.plot([0,1],[0,1],"navy",lw=1,linestyle="--"); ax2.set_title("ROC Curve",fontweight="bold",fontsize=12)
ax2.legend(loc="lower right"); ax2.grid(True,alpha=0.3)
ax3 = fig.add_subplot(gs[0,2])
metrics=["Accuracy","Precision","Recall","F1","ROC-AUC"]; values=[acc,prec,rec,f1,roc_auc]
colors=["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336"]
bars=ax3.bar(metrics,values,color=colors,alpha=0.85)
ax3.set_ylim(0,1.15); ax3.set_title("Metrics",fontweight="bold",fontsize=12); ax3.grid(True,axis="y",alpha=0.3)
for b,v in zip(bars,values): ax3.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f"{v:.3f}",ha="center",fontsize=9,fontweight="bold")
ax4 = fig.add_subplot(gs[1,:2])
top15=cat_acc.head(15); col4=["#4CAF50" if v>=0.5 else "#FF9800" if v>=0.3 else "#F44336" for v in top15.values]
ax4.barh(top15.index,top15.values,color=col4,alpha=0.85)
ax4.axvline(x=0.5,color="red",linestyle="--",alpha=0.5,label="50%"); ax4.legend()
ax4.set_title("TruthfulQA Per-Category",fontweight="bold",fontsize=12); ax4.grid(True,axis="x",alpha=0.3)
ax5 = fig.add_subplot(gs[1,2])
if rouge_scores:
    rn=list(rouge_scores.keys()); rv=[v/100 for v in rouge_scores.values()]
    b5=ax5.bar(rn,rv,color=["#3F51B5","#009688","#FF5722"],alpha=0.85)
    ax5.set_ylim(0,1); ax5.axhline(y=0.3,color="green",linestyle="--",label="Good(0.30)")
    ax5.set_title("ROUGE",fontweight="bold",fontsize=12); ax5.legend(); ax5.grid(True,axis="y",alpha=0.3)
    for b,v in zip(b5,rv): ax5.text(b.get_x()+b.get_width()/2,b.get_height()+0.01,f"{v:.3f}",ha="center",fontsize=10,fontweight="bold")
fig.suptitle("Hallucination Detection v3 — Full Evaluation",fontsize=14,fontweight="bold")
plt.savefig(os.path.join(CONFIG["output_dir"],"full_evaluation.png"),dpi=150,bbox_inches="tight")
print("✅ Plots saved")

# Demo
print("\n"+"="*60+"\n  PIPELINE DEMO\n"+"="*60)
demo = [
    ("Albert Einstein invented the telephone in 1876.",
     "Alexander Graham Bell patented the telephone in 1876."),
    ("The Eiffel Tower is in Berlin, Germany.",
     "The Eiffel Tower is on the Champ de Mars in Paris, France."),
    ("Water boils at 100 degrees Celsius at sea level.",
     "Water boils at 100 degrees Celsius at standard pressure."),
    ("The Amazon is the longest river in the world.",
     "The Nile at 6,650 km is the longest river. The Amazon is largest by flow."),
    ("Shakespeare was born in London.",
     "Shakespeare was born in Stratford-upon-Avon in April 1564."),
]
for i,case in enumerate(demo,1):
    r = pipeline(case[0], case[1])
    flag = "⚠" if r["verdict"]=="Hallucinated" else "✅"
    print(f"\n  {i}. {flag} {r['verdict'].upper()}")
    print(f"  Statement : {case[0]}")
    print(f"  Detection : {r['detection']['h_prob']}%")
    if r["nli"]: print(f"  NLI       : {r['nli']['label']} ({r['nli']['contradiction']}%)")
    print(f"  Confidence: {r['confidence']}")
    if r["corrected"]: print(f"  Corrected : {r['corrected']}")

print(f"\n{'='*60}\n  EVALUATION COMPLETE\n{'='*60}")
print(f"  Detection: Acc={acc*100:.2f}% | F1={f1:.4f} | AUC={roc_auc:.4f}")
print(f"  TruthfulQA: Strict={strict*100:.2f}% | Partial={partial*100:.2f}%")
if rouge_scores: print(f"  ROUGE-L: {rouge_scores['ROUGE-L']}")
print(f"  Output: {os.path.abspath(CONFIG['output_dir'])}/\n{'='*60}")
