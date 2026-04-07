"""
=============================================================
  Phase 4 — NLI Verification Layer Training
  Model  : cross-encoder/nli-roberta-base (fine-tuned)
  GPU    : RTX 3050 Mobile 4GB — Linux
  Input  : outputs/nli_train.csv
           outputs/nli_val.csv
           outputs/nli_test.csv
  Output : nli_model/best/
           nli_model/final/
           nli_model/results.txt
           nli_model/training_results.png
=============================================================
WHAT IS NLI?
  Natural Language Inference determines the relationship
  between two sentences:
    0 = Entailment   → evidence SUPPORTS the statement
    1 = Neutral      → evidence is unrelated to statement
    2 = Contradiction → evidence CONTRADICTS the statement

  In hallucination detection:
    Contradiction = strong signal that statement is hallucinated
    Entailment    = strong signal that statement is factual

HOW TO RUN:
  python train_nli.py

  Run this in a SECOND terminal while Phase 3 trains
  (they use different models and output folders)
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

# =============================================================
# CONFIGURATION
# =============================================================
CONFIG = {
    # Paths
    "train_path"        : "./outputs/nli_train.csv",
    "val_path"          : "./outputs/nli_val.csv",
    "test_path"         : "./outputs/nli_test.csv",
    "output_dir"        : "./nli_model",

    # Model — pre-trained NLI model, already knows entailment/contradiction
    # Fine-tuning on SNLI improves it further for our specific use case
    "model_name"        : "cross-encoder/nli-roberta-base",

    # Input length — 128 safe for 4GB VRAM
    "max_length"        : 128,

    # Batch size — NLI model is slightly smaller, 16 fits in 4GB
    # Reduce to 8 if you get out-of-memory errors
    "batch_size"        : 16,

    # Gradient accumulation — effective batch = 16x2 = 32
    "grad_accum"        : 2,

    # Training — 2 epochs enough for NLI fine-tuning
    # (model already pre-trained on NLI tasks)
    "epochs"            : 2,
    "learning_rate"     : 2e-5,
    "warmup_ratio"      : 0.1,
    "weight_decay"      : 0.01,
    "max_grad_norm"     : 1.0,

    # Limit samples — NLI train has 549,367 rows
    # 100,000 is enough for strong performance
    # Set to None to use all 549,367 (will take ~12+ hours)
    "max_train_samples" : 100000,

    # Linux supports multiple workers
    "num_workers"       : 4,

    "seed"              : 42,
}

# =============================================================
# STEP 1 — DEVICE SETUP
# =============================================================

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

print("=" * 60)
print("  Hallucination Detection — Phase 4 NLI Training")
print("=" * 60)

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n✅ GPU : {gpu_name}")
    print(f"   VRAM: {vram_gb:.1f} GB")
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
else:
    device = torch.device("cpu")
    print("\n⚠ GPU not found — running on CPU (very slow)")

print(f"   Device: {device}\n")

# =============================================================
# STEP 2 — LOAD DATA
# =============================================================

print("--- Loading NLI datasets ---")

df_train = pd.read_csv(CONFIG["train_path"], encoding="latin-1")
df_val   = pd.read_csv(CONFIG["val_path"],   encoding="latin-1")
df_test  = pd.read_csv(CONFIG["test_path"],  encoding="latin-1")

# Drop rows with label = -1 (no annotator consensus)
# Already done in preprocessing but double-checking here
df_train = df_train[df_train["nli_label"] != -1].reset_index(drop=True)
df_val   = df_val[df_val["nli_label"]     != -1].reset_index(drop=True)
df_test  = df_test[df_test["nli_label"]   != -1].reset_index(drop=True)

# Drop rows with missing values
df_train = df_train.dropna(subset=["premise", "hypothesis", "nli_label"]).reset_index(drop=True)
df_val   = df_val.dropna(subset=["premise", "hypothesis", "nli_label"]).reset_index(drop=True)
df_test  = df_test.dropna(subset=["premise", "hypothesis", "nli_label"]).reset_index(drop=True)

# Fill any remaining nulls
for df in [df_train, df_val, df_test]:
    df["premise"]   = df["premise"].fillna("").astype(str)
    df["hypothesis"]= df["hypothesis"].fillna("").astype(str)
    df["nli_label"] = df["nli_label"].astype(int)

# Limit training samples
if CONFIG["max_train_samples"] and len(df_train) > CONFIG["max_train_samples"]:
    df_train = df_train.sample(
        n            = CONFIG["max_train_samples"],
        random_state = CONFIG["seed"]
    ).reset_index(drop=True)
    print(f"   Training samples capped at: {CONFIG['max_train_samples']}")

print(f"   Train : {len(df_train)} samples")
print(f"   Val   : {len(df_val)} samples")
print(f"   Test  : {len(df_test)} samples")
print(f"   Label distribution (train):")
for label, name in [(0, "Entailment"), (1, "Neutral"), (2, "Contradiction")]:
    count = (df_train["nli_label"] == label).sum()
    print(f"     {label} - {name:<15}: {count}")
print()

# =============================================================
# STEP 3 — TOKENIZER
# =============================================================

print("--- Loading tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
print("   Tokenizer loaded\n")

# =============================================================
# STEP 4 — DATASET CLASS
# =============================================================

class NLIDataset(Dataset):
    """
    PyTorch Dataset for NLI task.
    Input format: [CLS] premise [SEP] hypothesis [SEP]
    Labels: 0=Entailment, 1=Neutral, 2=Contradiction
    """

    def __init__(self, dataframe, tokenizer, max_length):
        self.data      = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row        = self.data.iloc[idx]
        premise    = str(row["premise"])
        hypothesis = str(row["hypothesis"])

        encoding = self.tokenizer(
            premise,
            hypothesis,
            max_length     = self.max_len,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt"
        )

        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "label"          : torch.tensor(int(row["nli_label"]), dtype=torch.long)
        }


# Build datasets and loaders
train_dataset = NLIDataset(df_train, tokenizer, CONFIG["max_length"])
val_dataset   = NLIDataset(df_val,   tokenizer, CONFIG["max_length"])
test_dataset  = NLIDataset(df_test,  tokenizer, CONFIG["max_length"])

train_loader = DataLoader(
    train_dataset,
    batch_size         = CONFIG["batch_size"],
    shuffle            = True,
    num_workers        = CONFIG["num_workers"],
    pin_memory         = True if torch.cuda.is_available() else False,
    persistent_workers = True if CONFIG["num_workers"] > 0 else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size         = CONFIG["batch_size"] * 2,
    shuffle            = False,
    num_workers        = CONFIG["num_workers"],
    pin_memory         = True if torch.cuda.is_available() else False,
    persistent_workers = True if CONFIG["num_workers"] > 0 else False
)

test_loader = DataLoader(
    test_dataset,
    batch_size         = CONFIG["batch_size"] * 2,
    shuffle            = False,
    num_workers        = CONFIG["num_workers"],
    pin_memory         = True if torch.cuda.is_available() else False,
    persistent_workers = True if CONFIG["num_workers"] > 0 else False
)

print(f"--- DataLoaders ready ---")
print(f"   Train batches : {len(train_loader)}")
print(f"   Val batches   : {len(val_loader)}")
print(f"   Test batches  : {len(test_loader)}\n")

# =============================================================
# STEP 5 — LOAD MODEL
# =============================================================

print("--- Loading NLI model ---")
print("   (downloads ~500MB on first run, cached after)\n")

model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels = 3,       # 3 classes: entailment, neutral, contradiction
    use_cache  = False    # required for gradient checkpointing
)

model = model.to(device)
model.gradient_checkpointing_enable()

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters    : {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}\n")

# =============================================================
# STEP 6 — OPTIMIZER AND SCHEDULER
# =============================================================

optimizer = AdamW(
    model.parameters(),
    lr           = CONFIG["learning_rate"],
    weight_decay = CONFIG["weight_decay"]
)

total_steps  = (len(train_loader) // CONFIG["grad_accum"]) * CONFIG["epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = warmup_steps,
    num_training_steps = total_steps
)

print(f"--- Training schedule ---")
print(f"   Epochs          : {CONFIG['epochs']}")
print(f"   Total steps     : {total_steps}")
print(f"   Warmup steps    : {warmup_steps}")
print(f"   Effective batch : {CONFIG['batch_size'] * CONFIG['grad_accum']}")
print(f"   Learning rate   : {CONFIG['learning_rate']}\n")

# =============================================================
# STEP 7 — TRAINING FUNCTION
# =============================================================

def train_one_epoch(model, loader, optimizer, scheduler, device, grad_accum, epoch):
    model.train()
    total_loss = 0
    correct    = 0
    total      = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):

        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels
        )

        loss = outputs.loss / grad_accum
        loss.backward()

        total_loss += outputs.loss.item()

        preds    = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 100 == 0:
            avg_loss = total_loss / (step + 1)
            acc      = correct / total
            lr       = scheduler.get_last_lr()[0]
            print(f"   Epoch {epoch} | Step {step+1:>5}/{len(loader)} "
                  f"| Loss: {avg_loss:.4f} "
                  f"| Acc: {acc:.4f} "
                  f"| LR: {lr:.2e}")

            if torch.cuda.is_available():
                used       = torch.cuda.memory_allocated() / 1e9
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"             GPU memory: {used:.2f}/{total_vram:.1f} GB")

    return total_loss / len(loader), correct / total


# =============================================================
# STEP 8 — EVALUATION FUNCTION
# =============================================================

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
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
            preds       = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), all_labels, all_preds


# =============================================================
# STEP 9 — NLI PREDICTION FUNCTION
# =============================================================

def predict_nli(premise, hypothesis):
    """
    Predicts NLI relationship between premise and hypothesis.

    In hallucination detection:
      - Use evidence as premise
      - Use the statement to check as hypothesis
      - Contradiction = hallucination signal
      - Entailment    = factual signal

    Args:
        premise    : the evidence/context (what we know is true)
        hypothesis : the statement to verify

    Returns:
        dict with label and probabilities for all 3 classes
    """
    model.eval()

    encoding = tokenizer(
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    entailment_prob    = round(probs[0].item() * 100, 2)
    neutral_prob       = round(probs[1].item() * 100, 2)
    contradiction_prob = round(probs[2].item() * 100, 2)

    label_idx = torch.argmax(probs).item()
    label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

    return {
        "label"             : label_map[label_idx],
        "entailment_prob"   : entailment_prob,
        "neutral_prob"      : neutral_prob,
        "contradiction_prob": contradiction_prob,
        # Hallucination signal: high contradiction = likely hallucinated
        "hallucination_signal": "HIGH" if contradiction_prob > 50 else
                                "MEDIUM" if contradiction_prob > 30 else "LOW"
    }


# =============================================================
# STEP 10 — MAIN TRAINING LOOP
# =============================================================

if __name__ == "__main__":

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("\n" + "=" * 60)
    print("  NLI TRAINING STARTED")
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

        # Validate
        val_loss, true_labels, pred_labels = evaluate(model, val_loader, device)
        val_losses.append(val_loss)

        acc    = accuracy_score(true_labels, pred_labels)
        report = classification_report(
            true_labels, pred_labels,
            target_names = ["Entailment", "Neutral", "Contradiction"],
            output_dict  = True
        )
        f1 = report["macro avg"]["f1-score"]

        print(f"\n  Results — Epoch {epoch}:")
        print(f"    Train Loss     : {train_loss:.4f}")
        print(f"    Train Acc      : {train_acc:.4f}")
        print(f"    Val Loss       : {val_loss:.4f}")
        print(f"    Val Acc        : {acc:.4f}")
        print(f"    Macro F1       : {f1:.4f}")
        print(f"    Entailment     : precision={report['Entailment']['precision']:.3f}  recall={report['Entailment']['recall']:.3f}")
        print(f"    Neutral        : precision={report['Neutral']['precision']:.3f}  recall={report['Neutral']['recall']:.3f}")
        print(f"    Contradiction  : precision={report['Contradiction']['precision']:.3f}  recall={report['Contradiction']['recall']:.3f}")

        if f1 > best_f1:
            best_f1   = f1
            best_path = os.path.join(CONFIG["output_dir"], "best")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"\n  ✅ Best model saved — F1: {best_f1:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    final_path = os.path.join(CONFIG["output_dir"], "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Final model saved to: {final_path}")

    # ── FINAL TEST EVALUATION ──────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    _, true_labels, pred_labels = evaluate(model, test_loader, device)

    report_str = classification_report(
        true_labels, pred_labels,
        target_names = ["Entailment", "Neutral", "Contradiction"]
    )
    print(f"\n{report_str}")

    # Save results
    results_path = os.path.join(CONFIG["output_dir"], "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("NLI VERIFICATION MODEL — RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model       : {CONFIG['model_name']}\n")
        f.write(f"Max Length  : {CONFIG['max_length']}\n")
        f.write(f"Batch Size  : {CONFIG['batch_size']} (effective: {CONFIG['batch_size'] * CONFIG['grad_accum']})\n")
        f.write(f"Epochs      : {CONFIG['epochs']}\n")
        f.write(f"Train Size  : {len(df_train)}\n")
        f.write(f"Val Size    : {len(df_val)}\n")
        f.write(f"Test Size   : {len(df_test)}\n")
        f.write(f"Best F1     : {best_f1:.4f}\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report_str)
    print(f"✅ Results saved to: {results_path}")

    # ── PLOTS ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm   = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(
        cm, display_labels=["Entailment", "Neutral", "Contradiction"]
    )
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("NLI Confusion Matrix", fontsize=14, fontweight="bold")

    epochs_range = range(1, CONFIG["epochs"] + 1)
    axes[1].plot(epochs_range, train_losses, "b-o", label="Train Loss", linewidth=2)
    axes[1].plot(epochs_range, val_losses,   "r-o", label="Val Loss",   linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("NLI Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(CONFIG["output_dir"], "training_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved to: {plot_path}")

    # ── SAMPLE NLI PREDICTIONS ────────────────────────────
    print("\n--- Sample NLI Predictions ---")
    test_cases = [
        (
            "The Eiffel Tower is located in Paris, France.",
            "The Eiffel Tower is in London."
        ),
        (
            "Albert Einstein won the Nobel Prize in Physics in 1921.",
            "Einstein received the Nobel Prize for his work in physics."
        ),
        (
            "Water boils at 100 degrees Celsius at sea level.",
            "Water freezes at 100 degrees Celsius."
        ),
        (
            "The Amazon River flows through South America.",
            "The Amazon is a river in South America."
        ),
    ]

    for premise, hypothesis in test_cases:
        result = predict_nli(premise, hypothesis)
        flag   = "⚠" if result["label"] == "Contradiction" else "✅"
        print(f"\n  Premise    : {premise}")
        print(f"  Hypothesis : {hypothesis}")
        print(f"  {flag} Label : {result['label']} | "
              f"Entailment: {result['entailment_prob']}% | "
              f"Contradiction: {result['contradiction_prob']}% | "
              f"Hallucination Signal: {result['hallucination_signal']}")

    print("\n" + "=" * 60)
    print("  PHASE 4 COMPLETE")
    print(f"  Best model : {os.path.abspath(os.path.join(CONFIG['output_dir'], 'best'))}")
    print(f"  Final model: {os.path.abspath(final_path)}")
    print("=" * 60)
