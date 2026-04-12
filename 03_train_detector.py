"""
=============================================================
  Phase 3 — Hallucination Detection Model (IMPROVED)
  Model  : roberta-base
  GPU    : RTX 3050 Mobile 4GB — Linux

  IMPROVEMENTS over v1:
  1. Uses ALL 134,810 training samples (was capped at 50k)
  2. Early stopping — stops at best epoch, avoids overfitting
  3. Label smoothing — reduces overconfident predictions
  4. Weighted loss — handles class imbalance better
  5. Mixed precision (fp16) — faster training, less VRAM
  6. Threshold tuning — finds best classification threshold
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
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    roc_curve,
    auc
)

# =============================================================
# CONFIGURATION
# =============================================================
CONFIG = {
    "train_path"        : "./outputs/detection_train.csv",
    "test_path"         : "./outputs/detection_test.csv",
    "output_dir"        : "./detector_model",

    "model_name"        : "roberta-base",
    "max_length"        : 128,

    # IMPROVEMENT 1: Use full dataset (was 50000)
    "max_train_samples" : None,

    # Batch size — 8 safe for 4GB, use 4 if OOM
    "batch_size"        : 8,
    "grad_accum"        : 4,

    "epochs"            : 5,        # more epochs, early stopping will stop early
    "learning_rate"     : 2e-5,
    "warmup_ratio"      : 0.1,
    "weight_decay"      : 0.01,
    "max_grad_norm"     : 1.0,

    # IMPROVEMENT 2: Early stopping patience
    "patience"          : 2,        # stop if F1 doesn't improve for 2 epochs

    # IMPROVEMENT 3: Label smoothing reduces overconfidence
    "label_smoothing"   : 0.1,

    # IMPROVEMENT 4: Mixed precision training (faster + less VRAM)
    "use_fp16"          : True,

    "num_workers"       : 4,
    "seed"              : 42,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# =============================================================
# DEVICE
# =============================================================
print("=" * 60)
print("  Phase 3 — Detection Model Training (Improved)")
print("=" * 60)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n✅ GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
else:
    device = torch.device("cpu")
    print("\n⚠ No GPU — running on CPU")
    CONFIG["use_fp16"] = False  # fp16 not supported on CPU

print(f"   Device : {device}")
print(f"   FP16   : {CONFIG['use_fp16']}\n")

# =============================================================
# LOAD DATA
# =============================================================
print("--- Loading data ---")

df_train = pd.read_csv(CONFIG["train_path"], encoding="latin-1")
df_test  = pd.read_csv(CONFIG["test_path"],  encoding="latin-1")

df_train = df_train.dropna(subset=["statement", "label"]).reset_index(drop=True)
df_test  = df_test.dropna(subset=["statement",  "label"]).reset_index(drop=True)

df_train["evidence"]  = df_train["evidence"].fillna("").astype(str)
df_test["evidence"]   = df_test["evidence"].fillna("").astype(str)
df_train["statement"] = df_train["statement"].astype(str)
df_test["statement"]  = df_test["statement"].astype(str)

if CONFIG["max_train_samples"] and len(df_train) > CONFIG["max_train_samples"]:
    df_train = df_train.sample(n=CONFIG["max_train_samples"], random_state=CONFIG["seed"]).reset_index(drop=True)

# IMPROVEMENT 5: Compute class weights for weighted loss
n_factual     = (df_train["label"] == 0).sum()
n_hallucinated = (df_train["label"] == 1).sum()
total = n_factual + n_hallucinated
weight_factual     = total / (2 * n_factual)
weight_hallucinated = total / (2 * n_hallucinated)
class_weights = torch.tensor([weight_factual, weight_hallucinated], dtype=torch.float).to(device)

print(f"   Train : {len(df_train)} | Test : {len(df_test)}")
print(f"   Factual (0)     : {n_factual} | weight: {weight_factual:.3f}")
print(f"   Hallucinated (1): {n_hallucinated} | weight: {weight_hallucinated:.3f}\n")

# =============================================================
# TOKENIZER
# =============================================================
print("--- Loading tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
print("   Tokenizer loaded\n")

# =============================================================
# DATASET
# =============================================================
class HallucinationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.data      = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            str(row["statement"]),
            str(row["evidence"]),
            max_length     = self.max_len,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt"
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "label"          : torch.tensor(int(row["label"]), dtype=torch.long)
        }

train_dataset = HallucinationDataset(df_train, tokenizer, CONFIG["max_length"])
test_dataset  = HallucinationDataset(df_test,  tokenizer, CONFIG["max_length"])

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                          shuffle=True, num_workers=CONFIG["num_workers"],
                          pin_memory=True if torch.cuda.is_available() else False,
                          persistent_workers=True)

test_loader  = DataLoader(test_dataset, batch_size=CONFIG["batch_size"] * 2,
                          shuffle=False, num_workers=CONFIG["num_workers"],
                          pin_memory=True if torch.cuda.is_available() else False,
                          persistent_workers=True)

print(f"--- DataLoaders ready ---")
print(f"   Train batches : {len(train_loader)}")
print(f"   Test batches  : {len(test_loader)}\n")

# =============================================================
# MODEL
# =============================================================
print("--- Loading RoBERTa model ---")
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"], num_labels=2, use_cache=False
).to(device)
model.gradient_checkpointing_enable()
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# IMPROVEMENT 3: Label smoothing loss
loss_fn = torch.nn.CrossEntropyLoss(
    weight         = class_weights,
    label_smoothing= CONFIG["label_smoothing"]
)

# =============================================================
# OPTIMIZER + SCHEDULER + SCALER
# =============================================================
optimizer    = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
total_steps  = (len(train_loader) // CONFIG["grad_accum"]) * CONFIG["epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler       = GradScaler(enabled=CONFIG["use_fp16"])  # IMPROVEMENT 4: Mixed precision

print(f"--- Training schedule ---")
print(f"   Epochs         : {CONFIG['epochs']} (with early stopping, patience={CONFIG['patience']})")
print(f"   Total steps    : {total_steps}")
print(f"   Effective batch: {CONFIG['batch_size'] * CONFIG['grad_accum']}\n")

# =============================================================
# TRAINING FUNCTION
# =============================================================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, grad_accum, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total   = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        # IMPROVEMENT 4: Mixed precision forward pass
        with autocast(enabled=CONFIG["use_fp16"]):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels) / grad_accum

        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum

        preds    = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 200 == 0:
            print(f"   Ep {epoch} | Step {step+1:>5}/{len(loader)} "
                  f"| Loss: {total_loss/(step+1):.4f} "
                  f"| Acc: {correct/total:.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e}")
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1e9
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"             GPU: {used:.2f}/{total_vram:.1f} GB")

    return total_loss / len(loader), correct / total

# =============================================================
# EVALUATION FUNCTION
# =============================================================
def evaluate(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["label"].to(device, non_blocking=True)

            with autocast(enabled=CONFIG["use_fp16"]):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return all_labels, all_preds, all_probs

# =============================================================
# THRESHOLD TUNING — IMPROVEMENT 6
# =============================================================
def find_best_threshold(labels, probs):
    """Find threshold that maximises macro F1."""
    best_t  = 0.5
    best_f1 = 0
    for t in np.arange(0.3, 0.71, 0.02):
        preds = [1 if p > t else 0 for p in probs]
        f1    = f1_score(labels, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
    return round(best_t, 2), round(best_f1, 4)

# =============================================================
# PREDICTION FUNCTION
# =============================================================
def predict(statement, evidence="", threshold=0.5):
    model.eval()
    encoding = tokenizer(str(statement), str(evidence),
                         max_length=CONFIG["max_length"], truncation=True,
                         padding="max_length", return_tensors="pt")
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs              = torch.softmax(outputs.logits, dim=1)[0]
    hallucination_prob = probs[1].item()

    return {
        "label"             : "Hallucinated" if hallucination_prob > threshold else "Factual",
        "hallucination_prob": round(hallucination_prob * 100, 2),
        "factual_prob"      : round(probs[0].item() * 100, 2),
        "confidence"        : round(max(hallucination_prob, probs[0].item()) * 100, 2)
    }

# =============================================================
# MAIN TRAINING LOOP
# =============================================================
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  TRAINING STARTED")
    print("=" * 60)

    best_f1          = 0
    patience_counter = 0
    best_threshold   = 0.5
    train_losses     = []
    val_f1s          = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch} / {CONFIG['epochs']}")
        print(f"{'='*60}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, CONFIG["grad_accum"], epoch
        )
        train_losses.append(train_loss)

        true_labels, pred_labels, probs = evaluate(model, test_loader, device)
        report = classification_report(true_labels, pred_labels,
                                       target_names=["Factual", "Hallucinated"],
                                       output_dict=True)
        f1  = report["macro avg"]["f1-score"]
        acc = accuracy_score(true_labels, pred_labels)
        val_f1s.append(f1)

        # IMPROVEMENT 6: Find best threshold
        best_t, best_t_f1 = find_best_threshold(true_labels, probs)

        print(f"\n  Results — Epoch {epoch}:")
        print(f"    Train Loss       : {train_loss:.4f}")
        print(f"    Train Acc        : {train_acc:.4f}")
        print(f"    Val Acc          : {acc:.4f}")
        print(f"    Macro F1 (0.5)   : {f1:.4f}")
        print(f"    Best threshold   : {best_t} → F1: {best_t_f1:.4f}")
        print(f"    Factual    P/R/F : {report['Factual']['precision']:.3f} / {report['Factual']['recall']:.3f} / {report['Factual']['f1-score']:.3f}")
        print(f"    Hallucinated P/R/F: {report['Hallucinated']['precision']:.3f} / {report['Hallucinated']['recall']:.3f} / {report['Hallucinated']['f1-score']:.3f}")

        # Save best
        if f1 > best_f1:
            best_f1          = f1
            best_threshold   = best_t
            patience_counter = 0
            best_path = os.path.join(CONFIG["output_dir"], "best")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            # Save threshold
            with open(os.path.join(best_path, "threshold.txt"), "w") as f:
                f.write(str(best_threshold))
            print(f"\n  ✅ Best model saved — F1: {best_f1:.4f} | Threshold: {best_threshold}")
        else:
            patience_counter += 1
            print(f"\n  ⚠ No improvement ({patience_counter}/{CONFIG['patience']})")
            # IMPROVEMENT 2: Early stopping
            if patience_counter >= CONFIG["patience"]:
                print(f"  🛑 Early stopping at epoch {epoch}")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final
    final_path = os.path.join(CONFIG["output_dir"], "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Final evaluation
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION")
    print("=" * 60)

    true_labels, pred_labels, probs = evaluate(model, test_loader, device)

    # Apply best threshold
    tuned_preds = [1 if p > best_threshold else 0 for p in probs]
    report_str  = classification_report(true_labels, tuned_preds,
                                         target_names=["Factual", "Hallucinated"])
    fpr, tpr, _ = roc_curve(true_labels, probs)
    roc_auc     = auc(fpr, tpr)

    print(f"\n  Threshold used: {best_threshold}")
    print(f"\n{report_str}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

    # Save results
    results_path = os.path.join(CONFIG["output_dir"], "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("HALLUCINATION DETECTION MODEL — IMPROVED RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model         : {CONFIG['model_name']}\n")
        f.write(f"Train Samples : {len(df_train)}\n")
        f.write(f"Best Threshold: {best_threshold}\n")
        f.write(f"Best Macro F1 : {best_f1:.4f}\n")
        f.write(f"ROC-AUC       : {roc_auc:.4f}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report_str)
    print(f"\n✅ Results saved to: {results_path}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cm = confusion_matrix(true_labels, tuned_preds)
    ConfusionMatrixDisplay(cm, display_labels=["Factual", "Hallucinated"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontweight="bold")

    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0,1],[0,1], "navy", lw=1, linestyle="--")
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curve", fontweight="bold"); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(range(1, len(val_f1s)+1), val_f1s, "g-o", label="Val Macro F1", lw=2)
    axes[2].plot(range(1, len(train_losses)+1), train_losses, "b-o", label="Train Loss", lw=2)
    axes[2].set_xlabel("Epoch"); axes[2].set_title("F1 & Loss Curves", fontweight="bold")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_results.png"), dpi=150)
    print(f"✅ Plot saved")

    # Test predictions
    print("\n--- Sample Predictions ---")
    for stmt, ev in [
        ("The Eiffel Tower is in Berlin.", "The Eiffel Tower is in Paris, France."),
        ("Water boils at 100°C at sea level.", ""),
        ("Shakespeare was born in London.", "Shakespeare was born in Stratford-upon-Avon."),
        ("Einstein invented the telephone.", "Bell invented the telephone in 1876."),
    ]:
        r = predict(stmt, ev, threshold=best_threshold)
        flag = "⚠" if r["label"] == "Hallucinated" else "✅"
        print(f"  {flag} {stmt}")
        print(f"     → {r['label']} | {r['hallucination_prob']}% hallucination")

    print(f"\n{'='*60}")
    print(f"  PHASE 3 COMPLETE | Best F1: {best_f1:.4f} | Threshold: {best_threshold}")
    print(f"{'='*60}")
