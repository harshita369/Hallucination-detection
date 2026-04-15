"""
=============================================================
  Phase 3 — Hallucination Detection Model
  Model  : roberta-base
  GPU    : RTX 3050 Mobile 4GB — Linux

  ╔══════════════════════════════════════════════════════╗
  ║  ROOT CAUSE FIX — WHY OLD MODEL PREDICTED EVERYTHING ║
  ║  AS FACTUAL:                                         ║
  ║  The old model was trained on statement+evidence      ║
  ║  pairs. It learned to use evidence to decide.        ║
  ║  At inference time (predict.py) there is NO evidence ║
  ║  so it always defaulted to Factual.                  ║
  ║                                                      ║
  ║  FIX: Train on STATEMENT ONLY. The model must learn  ║
  ║  to detect hallucinations from language patterns,    ║
  ║  not from evidence. This makes it work at inference  ║
  ║  time without needing Wikipedia retrieval.           ║
  ╚══════════════════════════════════════════════════════╝
=============================================================
"""

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, roc_curve, auc
)

CONFIG = {
    "train_path"      : "./outputs/detection_train.csv",
    "test_path"       : "./outputs/detection_test.csv",
    "output_dir"      : "./detector_model",
    "model_name"      : "roberta-base",
    "max_length"      : 128,
    "max_train_samples": None,
    "batch_size"      : 8,
    "grad_accum"      : 4,
    "epochs"          : 5,
    "learning_rate"   : 2e-5,
    "warmup_ratio"    : 0.1,
    "weight_decay"    : 0.01,
    "max_grad_norm"   : 1.0,
    "patience"        : 2,
    "label_smoothing" : 0.1,
    "use_fp16"        : True,
    "num_workers"     : 4,
    "seed"            : 42,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

print("=" * 60)
print("  Phase 3 — Detection Model (STATEMENT-ONLY)")
print("=" * 60)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n✅ GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
else:
    device = torch.device("cpu")
    CONFIG["use_fp16"] = False
print(f"   Device: {device} | FP16: {CONFIG['use_fp16']}\n")

# ── Load data ─────────────────────────────────────────────
print("--- Loading data ---")
df_train = pd.read_csv(CONFIG["train_path"], encoding="latin-1")
df_test  = pd.read_csv(CONFIG["test_path"],  encoding="latin-1")
df_train = df_train.dropna(subset=["statement","label"]).reset_index(drop=True)
df_test  = df_test.dropna(subset=["statement","label"]).reset_index(drop=True)
df_train["statement"] = df_train["statement"].astype(str)
df_test["statement"]  = df_test["statement"].astype(str)

if CONFIG["max_train_samples"] and len(df_train) > CONFIG["max_train_samples"]:
    df_train = df_train.sample(n=CONFIG["max_train_samples"],
                                random_state=CONFIG["seed"]).reset_index(drop=True)

n0 = (df_train["label"] == 0).sum()
n1 = (df_train["label"] == 1).sum()
total = n0 + n1
class_weights = torch.tensor([total/(2*n0), total/(2*n1)], dtype=torch.float).to(device)

print(f"   Train : {len(df_train)} | Test : {len(df_test)}")
print(f"   Factual(0): {n0} | Hallucinated(1): {n1}")
print(f"   ⚠ Training on STATEMENT ONLY (no evidence — core fix)\n")

# ── Tokenizer ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

# ── Dataset — STATEMENT ONLY ──────────────────────────────
class DetectionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data = df.reset_index(drop=True)
        self.tok  = tokenizer
        self.max  = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # ONLY the statement — no evidence column used
        enc = self.tok(str(row["statement"]), max_length=self.max,
                       truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids"      : enc["input_ids"].squeeze(0),
            "attention_mask" : enc["attention_mask"].squeeze(0),
            "label"          : torch.tensor(int(row["label"]), dtype=torch.long)
        }

train_ds = DetectionDataset(df_train, tokenizer, CONFIG["max_length"])
test_ds  = DetectionDataset(df_test,  tokenizer, CONFIG["max_length"])

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                          num_workers=CONFIG["num_workers"], pin_memory=True,
                          persistent_workers=True)
test_loader  = DataLoader(test_ds, batch_size=CONFIG["batch_size"]*2, shuffle=False,
                          num_workers=CONFIG["num_workers"], pin_memory=True,
                          persistent_workers=True)

print(f"   Train batches: {len(train_loader)} | Test batches: {len(test_loader)}\n")

# ── Model ─────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"], num_labels=2, use_cache=False).to(device)
model.gradient_checkpointing_enable()
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights,
                                       label_smoothing=CONFIG["label_smoothing"])
optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                  weight_decay=CONFIG["weight_decay"])
total_steps  = (len(train_loader) // CONFIG["grad_accum"]) * CONFIG["epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler       = GradScaler(enabled=CONFIG["use_fp16"])

print(f"   Epochs: {CONFIG['epochs']} | Effective batch: {CONFIG['batch_size']*CONFIG['grad_accum']}\n")

# ── Train ─────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, device, grad_accum, ep):
    model.train()
    total_loss, correct, total = 0, 0, 0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        ids  = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        lbls = batch["label"].to(device, non_blocking=True)
        with autocast(enabled=CONFIG["use_fp16"]):
            out  = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(out.logits, lbls) / grad_accum
        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum
        correct    += (torch.argmax(out.logits,1) == lbls).sum().item()
        total      += lbls.size(0)
        if (step+1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            scaler.step(optimizer); scaler.update(); scheduler.step()
            optimizer.zero_grad()
        if (step+1) % 500 == 0:
            print(f"   Ep {ep} | Step {step+1:>5}/{len(loader)} "
                  f"| Loss: {total_loss/(step+1):.4f} | Acc: {correct/total:.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e}")
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated()/1e9
                tot  = torch.cuda.get_device_properties(0).total_memory/1e9
                print(f"             GPU: {used:.2f}/{tot:.1f} GB")
    return total_loss/len(loader), correct/total

def eval_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            lbls = batch["label"].to(device, non_blocking=True)
            with autocast(enabled=CONFIG["use_fp16"]):
                out = model(input_ids=ids, attention_mask=mask)
            probs = torch.softmax(out.logits, dim=1)
            all_preds.extend(torch.argmax(probs,1).cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())
    return all_labels, all_preds, all_probs

def find_threshold(labels, probs):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.3, 0.71, 0.02):
        preds = [1 if p > t else 0 for p in probs]
        f1 = f1_score(labels, preds, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(best_t, 2), round(best_f1, 4)

# ── Main training loop ────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TRAINING STARTED (Statement-Only Mode)")
    print("=" * 60)

    best_f1, patience_counter, best_threshold = 0, 0, 0.5
    train_losses, val_f1s = [], []

    for epoch in range(1, CONFIG["epochs"]+1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch} / {CONFIG['epochs']}")
        print(f"{'='*60}")

        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler,
                                       scaler, device, CONFIG["grad_accum"], epoch)
        train_losses.append(tr_loss)

        labels, preds, probs = eval_model(model, test_loader, device)
        rep = classification_report(labels, preds,
                                    target_names=["Factual","Hallucinated"],
                                    output_dict=True)
        f1  = rep["macro avg"]["f1-score"]
        acc = accuracy_score(labels, preds)
        val_f1s.append(f1)
        best_t, best_t_f1 = find_threshold(labels, probs)

        print(f"\n  Results — Epoch {epoch}:")
        print(f"    Train Loss        : {tr_loss:.4f}")
        print(f"    Val Acc           : {acc:.4f}")
        print(f"    Macro F1          : {f1:.4f}")
        print(f"    Best threshold    : {best_t} → F1: {best_t_f1:.4f}")
        print(f"    Factual    P/R/F  : {rep['Factual']['precision']:.3f}"
              f"/{rep['Factual']['recall']:.3f}/{rep['Factual']['f1-score']:.3f}")
        print(f"    Hallucinated P/R/F: {rep['Hallucinated']['precision']:.3f}"
              f"/{rep['Hallucinated']['recall']:.3f}/{rep['Hallucinated']['f1-score']:.3f}")

        if f1 > best_f1:
            best_f1, best_threshold, patience_counter = f1, best_t, 0
            bp = os.path.join(CONFIG["output_dir"], "best")
            model.save_pretrained(bp); tokenizer.save_pretrained(bp)
            with open(os.path.join(bp, "threshold.txt"), "w") as f:
                f.write(str(best_threshold))
            print(f"\n  ✅ Best model saved — F1: {best_f1:.4f} | Threshold: {best_threshold}")
        else:
            patience_counter += 1
            print(f"\n  ⚠ No improvement ({patience_counter}/{CONFIG['patience']})")
            if patience_counter >= CONFIG["patience"]:
                print("  🛑 Early stopping"); break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    fp = os.path.join(CONFIG["output_dir"], "final")
    model.save_pretrained(fp); tokenizer.save_pretrained(fp)

    # Final evaluation
    print("\n" + "="*60)
    print("  FINAL EVALUATION")
    print("="*60)
    labels, preds, probs = eval_model(model, test_loader, device)
    tuned = [1 if p > best_threshold else 0 for p in probs]
    rep_str = classification_report(labels, tuned, target_names=["Factual","Hallucinated"])
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    print(f"\n  Threshold: {best_threshold}\n{rep_str}\n  ROC-AUC: {roc_auc:.4f}")

    with open(os.path.join(CONFIG["output_dir"], "results.txt"), "w") as f:
        f.write("HALLUCINATION DETECTION MODEL — STATEMENT-ONLY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model         : {CONFIG['model_name']}\n")
        f.write(f"Training Mode : Statement-Only (no evidence)\n")
        f.write(f"Train Samples : {len(df_train)}\n")
        f.write(f"Threshold     : {best_threshold}\n")
        f.write(f"Best F1       : {best_f1:.4f}\n")
        f.write(f"ROC-AUC       : {roc_auc:.4f}\n\n")
        f.write(rep_str)
    print("✅ Results saved")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    cm = confusion_matrix(labels, tuned)
    ConfusionMatrixDisplay(cm, display_labels=["Factual","Hallucinated"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontweight="bold")
    axes[1].plot(fpr, tpr, "darkorange", lw=2, label=f"AUC={roc_auc:.3f}")
    axes[1].plot([0,1],[0,1],"navy",lw=1,linestyle="--")
    axes[1].set_title("ROC Curve", fontweight="bold"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(range(1,len(val_f1s)+1), val_f1s, "g-o", label="Val F1", lw=2)
    axes[2].plot(range(1,len(train_losses)+1), train_losses, "b-o",
                 label="Train Loss", lw=2)
    axes[2].set_title("F1 & Loss", fontweight="bold")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_results.png"), dpi=150)
    print("✅ Plot saved")

    # Inference test — works WITHOUT evidence
    print("\n--- Inference Test (No Evidence Needed) ---")
    model.eval()
    tests = [
        ("Milky Way galaxy is the name of a planet.", True),
        ("1 liter is equal to 1200 ml.",              True),
        ("1 km is equal to 1 liter.",                 True),
        ("Andromeda Galaxy is the galaxy we live in.", True),
        ("Water boils at 100°C at sea level.",         False),
        ("Paris is the capital of France.",            False),
        ("New York is a continent.",                   True),
        ("Einstein invented the telephone.",           True),
        ("Shakespeare was born in London.",            True),
        ("The Nile is the longest river.",             False),
    ]
    for stmt, is_hall in tests:
        enc = tokenizer(stmt, max_length=CONFIG["max_length"],
                        truncation=True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            with autocast(enabled=CONFIG["use_fp16"]):
                out = model(input_ids=enc["input_ids"].to(device),
                            attention_mask=enc["attention_mask"].to(device))
        prob = torch.softmax(out.logits, dim=1)[0][1].item()
        label    = "Hallucinated" if prob > best_threshold else "Factual"
        expected = "Hallucinated" if is_hall else "Factual"
        mark = "✅" if label == expected else "❌"
        print(f"  {mark} {stmt}\n       → {label} | {prob*100:.1f}%\n")

    print(f"{'='*60}")
    print(f"  PHASE 3 COMPLETE | Best F1: {best_f1:.4f} | Threshold: {best_threshold}")
    print(f"{'='*60}")
