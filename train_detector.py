"""
=============================================================
  Phase 3 — Hallucination Detection Model Training
  Model  : RoBERTa-base (fine-tuned classifier)
  GPU    : RTX 3050 Mobile 4GB VRAM (Linux)
  Input  : outputs/detection_train.csv
           outputs/detection_test.csv
  Output : detector_model/final/  (saved model)
           detector_model/results.txt (metrics)

  FIXES APPLIED (vs original Windows version):
  - use_cache=False passed to model (silences warning)
  - predict() defined before if __name__ block
  - No duplicate if __name__ blocks
  - num_workers=4 (Linux supports forking, Windows needed 0)
  - pin_memory=True only when GPU is available
  - persistent_workers=True (faster data loading on Linux)
  - non_blocking=True on .to(device) (faster GPU transfers)
  - gradient_checkpointing_enable() — saves ~40% VRAM
=============================================================
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
    accuracy_score
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =============================================================
# STEP 1 — CONFIGURATION
# All settings in one place — easy to adjust
# =============================================================

CONFIG = {
    # Paths
    "train_path"        : "./outputs/detection_train.csv",
    "test_path"         : "./outputs/detection_test.csv",
    "output_dir"        : "./detector_model",

    # Model
    "model_name"        : "roberta-base",
    "max_length"        : 128,          # 256 needs ~8GB VRAM, 128 fits in 4GB

    # Training — optimized for RTX 3050 4GB
    "batch_size"        : 8,            # safe limit for 4GB — do NOT increase
    "grad_accum"        : 4,            # effective batch = 8 x 4 = 32
    "epochs"            : 3,            # standard for BERT fine-tuning
    "learning_rate"     : 2e-5,         # standard LR for BERT fine-tuning
    "warmup_ratio"      : 0.1,          # 10% of steps for LR warmup
    "max_train_samples" : 50000,        # safe starting point — increase to 134810 later
    "weight_decay"      : 0.01,

    # Linux-specific optimizations
    "num_workers"       : 4,            # Linux supports forking — 4 workers is safe
    "seed"              : 42,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# =============================================================
# STEP 2 — SET DEVICE
# =============================================================

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

if torch.cuda.is_available():
    device    = torch.device("cuda")
    pin_mem   = True    # pin_memory=True only works with CUDA — faster GPU transfers
    print(f"\n✅ GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Helps with VRAM fragmentation on 4GB cards
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
else:
    device  = torch.device("cpu")
    pin_mem = False     # pin_memory must be False on CPU
    print("⚠ No GPU found — running on CPU (will be slow)")

print(f"   Device: {device}\n")

# =============================================================
# STEP 3 — LOAD DATA
# =============================================================

print("--- Loading datasets ---")

df_train = pd.read_csv(CONFIG["train_path"], encoding="latin-1")
df_test  = pd.read_csv(CONFIG["test_path"],  encoding="latin-1")

# Drop rows with missing statement or label
df_train = df_train.dropna(subset=["statement", "label"]).reset_index(drop=True)
df_test  = df_test.dropna(subset=["statement",  "label"]).reset_index(drop=True)

# Fill missing evidence with empty string
df_train["evidence"] = df_train["evidence"].fillna("").astype(str)
df_test["evidence"]  = df_test["evidence"].fillna("").astype(str)

# Ensure string types
df_train["statement"] = df_train["statement"].astype(str)
df_test["statement"]  = df_test["statement"].astype(str)

# Cap training samples for VRAM safety
if CONFIG["max_train_samples"] and len(df_train) > CONFIG["max_train_samples"]:
    df_train = df_train.sample(
        n=CONFIG["max_train_samples"],
        random_state=CONFIG["seed"]
    ).reset_index(drop=True)

print(f"   Train samples : {len(df_train)}")
print(f"   Test samples  : {len(df_test)}")
print(f"   Factual (0)      : {(df_train['label']==0).sum()}")
print(f"   Hallucinated (1) : {(df_train['label']==1).sum()}\n")

# =============================================================
# STEP 4 — TOKENIZER
# =============================================================

print("--- Loading tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
print("   Tokenizer loaded\n")

# =============================================================
# STEP 5 — DATASET CLASS
# =============================================================

class HallucinationDataset(Dataset):
    """
    Custom PyTorch Dataset.
    Each sample returns: input_ids, attention_mask, label
    """

    def __init__(self, dataframe, tokenizer, max_length):
        self.data      = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        statement = str(row["statement"])
        evidence  = str(row["evidence"])

        # Tokenize statement + evidence together
        # Format: [CLS] statement [SEP] evidence [SEP]
        encoding = self.tokenizer(
            statement,
            evidence,
            max_length     = self.max_len,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt"
        )

        return {
            # squeeze() removes the extra batch dim: [1, 128] → [128]
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "label"          : torch.tensor(int(row["label"]), dtype=torch.long)
        }


train_dataset = HallucinationDataset(df_train, tokenizer, CONFIG["max_length"])
test_dataset  = HallucinationDataset(df_test,  tokenizer, CONFIG["max_length"])

train_loader = DataLoader(
    train_dataset,
    batch_size        = CONFIG["batch_size"],
    shuffle           = True,
    num_workers       = CONFIG["num_workers"],   # 4 workers on Linux (safe)
    pin_memory        = pin_mem,                 # True only when CUDA available
    persistent_workers= True                     # keeps workers alive between epochs
)

test_loader = DataLoader(
    test_dataset,
    batch_size        = CONFIG["batch_size"] * 2,  # larger batch OK for eval (no gradients)
    shuffle           = False,
    num_workers       = CONFIG["num_workers"],
    pin_memory        = pin_mem,
    persistent_workers= True
)

print(f"--- DataLoaders ready ---")
print(f"   Train batches : {len(train_loader)}")
print(f"   Test batches  : {len(test_loader)}\n")

# =============================================================
# STEP 6 — LOAD MODEL
# =============================================================

print("--- Loading RoBERTa model ---")
print("   (downloading ~500MB on first run — cached after)\n")

model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels  = 2,       # binary: 0=factual, 1=hallucinated
    use_cache   = False    # FIX: suppresses warning about gradient checkpointing
)

model = model.to(device)

# Gradient checkpointing — trades compute for memory
# Reduces VRAM usage by ~40% — essential for 4GB GPU
model.gradient_checkpointing_enable()

total_params    = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total params   : {total_params:,}")
print(f"   Trainable params: {trainable_params:,}\n")

# =============================================================
# STEP 7 — OPTIMIZER AND SCHEDULER
# =============================================================

optimizer = AdamW(
    model.parameters(),
    lr           = CONFIG["learning_rate"],
    weight_decay = CONFIG["weight_decay"]
)

total_steps  = (len(train_loader) // CONFIG["grad_accum"]) * CONFIG["epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

# Linear warmup then linear decay to 0
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = warmup_steps,
    num_training_steps = total_steps
)

print(f"--- Training schedule ---")
print(f"   Total steps     : {total_steps}")
print(f"   Warmup steps    : {warmup_steps}")
print(f"   Effective batch : {CONFIG['batch_size'] * CONFIG['grad_accum']}")
print(f"   Learning rate   : {CONFIG['learning_rate']}\n")

# =============================================================
# STEP 8 — TRAINING FUNCTION
# =============================================================

def train_one_epoch(model, loader, optimizer, scheduler, device, grad_accum, epoch):
    """One full pass through training data. Returns average loss."""
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):

        # non_blocking=True: GPU transfer overlaps with computation
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        # Forward pass
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels   # model computes loss when labels provided
        )

        loss = outputs.loss / grad_accum   # scale loss for gradient accumulation
        loss.backward()

        total_loss += outputs.loss.item()   # log unscaled loss

        preds    = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        # Update weights every grad_accum steps
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 200 == 0:
            print(f"   Ep {epoch} | Step {step+1:>5}/{len(loader)} "
                  f"| Loss: {outputs.loss.item():.4f} "
                  f"| Acc: {correct/total:.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e}")
            if torch.cuda.is_available():
                used  = torch.cuda.memory_allocated() / 1e9
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"             GPU Memory: {used:.2f}/{total_vram:.1f} GB")

    return total_loss / len(loader), correct / total

# =============================================================
# STEP 9 — EVALUATION FUNCTION
# =============================================================

def evaluate(model, loader, device):
    """Evaluate model on test set. Returns loss, true labels, predicted labels."""
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():   # no gradients needed during evaluation
        for batch in loader:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["label"].to(device, non_blocking=True)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels
            )

            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), all_labels, all_preds

# =============================================================
# STEP 10 — PREDICTION FUNCTION
# Defined here (before if __name__) so it's always available
# =============================================================

def predict(statement, evidence=""):
    """
    Predict whether a statement is factual or hallucinated.
    Args:
        statement : text to check
        evidence  : optional supporting context (can be empty)
    Returns:
        dict with label, hallucination_prob, factual_prob, confidence
    """
    model.eval()

    encoding = tokenizer(
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    hallucination_prob = probs[1].item()
    factual_prob       = probs[0].item()

    return {
        "label"             : "Hallucinated" if hallucination_prob > 0.5 else "Factual",
        "hallucination_prob": round(hallucination_prob * 100, 2),
        "factual_prob"      : round(factual_prob * 100, 2),
        "confidence"        : round(max(hallucination_prob, factual_prob) * 100, 2)
    }

# =============================================================
# STEP 11 — MAIN TRAINING LOOP
# Everything below here only runs when script is executed directly
# (not when imported). This is required for Linux multiprocessing.
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  TRAINING STARTED")
    print("=" * 60)

    best_f1      = 0
    train_losses = []
    val_losses   = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch} / {CONFIG['epochs']}")
        print(f"{'='*60}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, CONFIG["grad_accum"], epoch
        )
        train_losses.append(train_loss)

        # Evaluate
        val_loss, true_labels, pred_labels = evaluate(model, test_loader, device)
        val_losses.append(val_loss)

        # Metrics
        acc    = accuracy_score(true_labels, pred_labels)
        report = classification_report(
            true_labels, pred_labels,
            target_names=["Factual", "Hallucinated"],
            output_dict=True
        )
        f1 = report["macro avg"]["f1-score"]

        print(f"\n  Results — Epoch {epoch}:")
        print(f"    Train Loss : {train_loss:.4f}")
        print(f"    Train Acc  : {train_acc:.4f}")
        print(f"    Val Loss   : {val_loss:.4f}")
        print(f"    Val Acc    : {acc:.4f}")
        print(f"    Macro F1   : {f1:.4f}")
        print(f"    Factual      P/R/F: {report['Factual']['precision']:.3f}"
              f" / {report['Factual']['recall']:.3f}"
              f" / {report['Factual']['f1-score']:.3f}")
        print(f"    Hallucinated P/R/F: {report['Hallucinated']['precision']:.3f}"
              f" / {report['Hallucinated']['recall']:.3f}"
              f" / {report['Hallucinated']['f1-score']:.3f}")

        # Save best model
        if f1 > best_f1:
            best_f1   = f1
            best_path = os.path.join(CONFIG["output_dir"], "best")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"\n  ✅ Best model saved — F1: {best_f1:.4f}")

        # Free VRAM between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    final_path = os.path.join(CONFIG["output_dir"], "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Final model saved: {final_path}")

    # ─── Final evaluation ────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    _, true_labels, pred_labels = evaluate(model, test_loader, device)

    report_str = classification_report(
        true_labels, pred_labels,
        target_names=["Factual", "Hallucinated"]
    )
    print(f"\n{report_str}")

    # Save results to file
    results_path = os.path.join(CONFIG["output_dir"], "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("HALLUCINATION DETECTION MODEL — RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model       : {CONFIG['model_name']}\n")
        f.write(f"Max Length  : {CONFIG['max_length']}\n")
        f.write(f"Batch Size  : {CONFIG['batch_size']} "
                f"(effective: {CONFIG['batch_size']*CONFIG['grad_accum']})\n")
        f.write(f"Epochs      : {CONFIG['epochs']}\n")
        f.write(f"Train Size  : {len(df_train)}\n")
        f.write(f"Test Size   : {len(df_test)}\n")
        f.write(f"Best F1     : {best_f1:.4f}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report_str)
    print(f"✅ Results saved to: {results_path}")

    # ─── Plots ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm   = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Factual", "Hallucinated"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    axes[1].plot(range(1, CONFIG["epochs"]+1), train_losses, "b-o", label="Train Loss", linewidth=2)
    axes[1].plot(range(1, CONFIG["epochs"]+1), val_losses,   "r-o", label="Val Loss",   linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(CONFIG["output_dir"], "training_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved to: {plot_path}")

    # ─── Test predictions ─────────────────────────────────
    print("\n--- Sample Predictions ---")
    test_cases = [
        ("The Eiffel Tower is located in Paris, France.", ""),
        ("Albert Einstein invented the telephone.",       ""),
        ("The Great Wall of China is visible from space.", ""),
        ("Water boils at 100 degrees Celsius at sea level.", ""),
        ("Shakespeare was born in London.",               ""),
    ]

    for statement, evidence in test_cases:
        result = predict(statement, evidence)
        flag   = "⚠" if result["label"] == "Hallucinated" else "✅"
        print(f"\n  {flag} {statement}")
        print(f"       → {result['label']} ({result['hallucination_prob']}% hallucination)")

    print("\n" + "=" * 60)
    print("  ✅ PHASE 3 COMPLETE")
    print(f"  Best F1    : {best_f1:.4f}")
    print(f"  Model path : {os.path.abspath(final_path)}")
    print("=" * 60)
