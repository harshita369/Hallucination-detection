"""
=============================================================
  Phase 6 — Correction Engine (IMPROVED)
  Model  : google/flan-t5-small
  GPU    : RTX 3050 Mobile 4GB — Linux

  IMPROVEMENTS over v1:
  1. Better prompt — explicit full-sentence instruction
  2. Lower learning rate (1e-4 vs 3e-4) — fixes flat val loss
  3. All 20,000 samples used (was 10,000)
  4. length_penalty + min_length — forces full sentence output
  5. Early stopping on ROUGE-L
  6. Mixed precision (fp16) training
  7. Cosine LR scheduler — better convergence than linear
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
    T5Tokenizer,
    T5ForConditionalGeneration,
    get_cosine_schedule_with_warmup   # IMPROVEMENT 7
)
from torch.optim import AdamW

try:
    from evaluate import load as load_metric
    rouge = load_metric("rouge")
    ROUGE_AVAILABLE = True
except:
    ROUGE_AVAILABLE = False
    print("⚠ pip install evaluate rouge-score")

# =============================================================
# CONFIGURATION
# =============================================================
CONFIG = {
    "train_path"        : "./outputs/correction_train.csv",
    "output_dir"        : "./corrector_model",

    # flan-t5-small proved better than base for this task
    "model_name"        : "google/flan-t5-small",

    "max_input_length"  : 256,
    "max_output_length" : 128,

    "batch_size"        : 8,
    "grad_accum"        : 4,          # effective batch = 32

    "epochs"            : 6,          # more epochs with early stopping
    # IMPROVEMENT 2: Lower LR — fixes flat validation loss
    "learning_rate"     : 1e-4,       # was 3e-4
    "warmup_ratio"      : 0.1,
    "weight_decay"      : 0.01,
    "max_grad_norm"     : 1.0,

    # IMPROVEMENT 5: Early stopping
    "patience"          : 2,

    # IMPROVEMENT 3: Use all 20,000 samples
    "max_train_samples" : None,       # was 10000

    "val_split"         : 0.1,

    # IMPROVEMENT 6: Mixed precision
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
print("  Phase 6 — Correction Engine (Improved)")
print("=" * 60)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n✅ GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
else:
    device = torch.device("cpu")
    CONFIG["use_fp16"] = False
    print("\n⚠ No GPU — running on CPU")

print(f"   Device: {device}\n")

# =============================================================
# LOAD DATA
# =============================================================
print("--- Loading correction dataset ---")

df = pd.read_csv(CONFIG["train_path"], encoding="latin-1")
df = df.dropna(subset=["wrong_statement", "correct_statement"]).reset_index(drop=True)
df["evidence"]          = df["evidence"].fillna("").astype(str)
df["wrong_statement"]   = df["wrong_statement"].astype(str)
df["correct_statement"] = df["correct_statement"].astype(str)

# Filter out noise — correct statement must be meaningful
df = df[df["correct_statement"].str.len() > 5].reset_index(drop=True)
df = df[df["wrong_statement"].str.len() > 5].reset_index(drop=True)

if CONFIG["max_train_samples"] and len(df) > CONFIG["max_train_samples"]:
    df = df.sample(n=CONFIG["max_train_samples"], random_state=CONFIG["seed"]).reset_index(drop=True)

val_size   = int(len(df) * CONFIG["val_split"])
df_train   = df.iloc[:len(df)-val_size].reset_index(drop=True)
df_val     = df.iloc[len(df)-val_size:].reset_index(drop=True)

print(f"   Train : {len(df_train)} | Val : {len(df_val)}")
print(f"   Sample pair:")
print(f"     Wrong  : {df_train['wrong_statement'].iloc[0][:80]}...")
print(f"     Correct: {df_train['correct_statement'].iloc[0][:80]}...\n")

# =============================================================
# TOKENIZER
# =============================================================
print("--- Loading tokenizer ---")
tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
print("   Tokenizer loaded\n")

# =============================================================
# PROMPT BUILDER — IMPROVEMENT 1
# =============================================================
def build_input_prompt(wrong_statement, evidence):
    """
    IMPROVED prompt — more explicit instruction forces
    the model to produce a full corrected sentence.
    """
    if evidence and evidence.strip():
        return (
            f"Given the following evidence: '{evidence}' "
            f"The statement '{wrong_statement}' is factually incorrect. "
            f"Write a complete corrected factual sentence:"
        )
    else:
        return (
            f"The following statement is factually incorrect: '{wrong_statement}' "
            f"Write a complete corrected factual sentence:"
        )

# =============================================================
# DATASET
# =============================================================
class CorrectionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_len, max_output_len):
        self.data           = dataframe.reset_index(drop=True)
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        input_text  = build_input_prompt(str(row["wrong_statement"]), str(row["evidence"]))
        target_text = str(row["correct_statement"])

        input_enc = self.tokenizer(
            input_text,
            max_length     = self.max_input_len,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt"
        )
        target_enc = self.tokenizer(
            text_target    = target_text,
            max_length     = self.max_output_len,
            truncation     = True,
            padding        = "max_length",
            return_tensors = "pt"
        )

        labels = target_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids"      : input_enc["input_ids"].squeeze(0),
            "attention_mask" : input_enc["attention_mask"].squeeze(0),
            "labels"         : labels
        }

train_dataset = CorrectionDataset(df_train, tokenizer, CONFIG["max_input_length"], CONFIG["max_output_length"])
val_dataset   = CorrectionDataset(df_val,   tokenizer, CONFIG["max_input_length"], CONFIG["max_output_length"])

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                          shuffle=True, num_workers=CONFIG["num_workers"],
                          pin_memory=torch.cuda.is_available(), persistent_workers=True)
val_loader   = DataLoader(val_dataset, batch_size=CONFIG["batch_size"]*2,
                          shuffle=False, num_workers=CONFIG["num_workers"],
                          pin_memory=torch.cuda.is_available(), persistent_workers=True)

print(f"--- DataLoaders ready ---")
print(f"   Train batches : {len(train_loader)}")
print(f"   Val batches   : {len(val_loader)}\n")

# =============================================================
# MODEL
# =============================================================
print("--- Loading Flan-T5-small model ---")
model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"]).to(device)
model.gradient_checkpointing_enable()
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# =============================================================
# OPTIMIZER + SCHEDULER
# =============================================================
optimizer    = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
total_steps  = (len(train_loader) // CONFIG["grad_accum"]) * CONFIG["epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])

# IMPROVEMENT 7: Cosine scheduler — smoother LR decay
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler    = GradScaler(enabled=CONFIG["use_fp16"])

print(f"--- Training schedule ---")
print(f"   Epochs         : {CONFIG['epochs']} (early stopping patience={CONFIG['patience']})")
print(f"   Effective batch: {CONFIG['batch_size'] * CONFIG['grad_accum']}")
print(f"   Learning rate  : {CONFIG['learning_rate']}\n")

# =============================================================
# TRAINING FUNCTION
# =============================================================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, grad_accum, epoch):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)

        with autocast(enabled=CONFIG["use_fp16"]):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss / grad_accum

        scaler.scale(loss).backward()
        total_loss += outputs.loss.item()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 100 == 0:
            print(f"   Ep {epoch} | Step {step+1:>4}/{len(loader)} "
                  f"| Loss: {total_loss/(step+1):.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e}")
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1e9
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"             GPU: {used:.2f}/{total_vram:.1f} GB")

    return total_loss / len(loader)

# =============================================================
# VALIDATION FUNCTION
# =============================================================
def validate(model, loader, device, num_generate=200):
    model.eval()
    total_loss    = 0
    generated     = []
    references    = []
    generate_count = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)

            with autocast(enabled=CONFIG["use_fp16"]):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            if generate_count < num_generate:
                # IMPROVEMENT 4: length_penalty + min_length
                gen_ids = model.generate(
                    input_ids            = input_ids,
                    attention_mask       = attention_mask,
                    max_length           = CONFIG["max_output_length"],
                    min_length           = 8,       # forces complete sentences
                    num_beams            = 4,
                    length_penalty       = 1.5,     # penalizes very short outputs
                    early_stopping       = True,
                    no_repeat_ngram_size = 3
                )

                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                label_ids = labels.clone()
                label_ids[label_ids == -100] = tokenizer.pad_token_id
                refs  = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

                generated.extend(preds)
                references.extend(refs)
                generate_count += len(input_ids)

    avg_loss = total_loss / len(loader)
    rouge_scores = {}
    if ROUGE_AVAILABLE and generated:
        scores = rouge.compute(predictions=generated, references=references)
        rouge_scores = {
            "rouge1": round(scores["rouge1"] * 100, 2),
            "rouge2": round(scores["rouge2"] * 100, 2),
            "rougeL": round(scores["rougeL"] * 100, 2),
        }
    return avg_loss, rouge_scores, generated[:3], references[:3]

# =============================================================
# CORRECTION FUNCTION
# =============================================================
def correct_statement(wrong_statement, evidence=""):
    model.eval()
    input_text = build_input_prompt(wrong_statement, evidence)
    inputs = tokenizer(input_text, max_length=CONFIG["max_input_length"],
                       truncation=True, padding="max_length", return_tensors="pt")
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            max_length           = CONFIG["max_output_length"],
            min_length           = 8,        # IMPROVEMENT 4
            num_beams            = 4,
            length_penalty       = 1.5,      # IMPROVEMENT 4
            early_stopping       = True,
            no_repeat_ngram_size = 3,
            do_sample            = False
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  CORRECTION ENGINE TRAINING STARTED")
    print("=" * 60)

    best_rouge        = 0
    patience_counter  = 0
    train_losses      = []
    val_losses        = []
    rouge_l_history   = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch} / {CONFIG['epochs']}")
        print(f"{'='*60}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, CONFIG["grad_accum"], epoch
        )
        train_losses.append(train_loss)

        val_loss, rouge_scores, sample_preds, sample_refs = validate(model, val_loader, device)
        val_losses.append(val_loss)

        current_rouge = rouge_scores.get("rougeL", 0) if rouge_scores else 0
        rouge_l_history.append(current_rouge)

        print(f"\n  Results — Epoch {epoch}:")
        print(f"    Train Loss : {train_loss:.4f}")
        print(f"    Val Loss   : {val_loss:.4f}")
        if rouge_scores:
            print(f"    ROUGE-1    : {rouge_scores.get('rouge1')}")
            print(f"    ROUGE-2    : {rouge_scores.get('rouge2')}")
            print(f"    ROUGE-L    : {rouge_scores.get('rougeL')}")

        print(f"\n  Sample corrections:")
        for pred, ref in zip(sample_preds, sample_refs):
            print(f"    Generated: {pred[:100]}")
            print(f"    Reference: {ref[:100]}")
            print()

        # IMPROVEMENT 5: Early stopping on ROUGE-L
        if current_rouge > best_rouge:
            best_rouge       = current_rouge
            patience_counter = 0
            best_path = os.path.join(CONFIG["output_dir"], "best")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"  ✅ Best model saved — ROUGE-L: {best_rouge:.2f}")
        else:
            patience_counter += 1
            print(f"  ⚠ No ROUGE-L improvement ({patience_counter}/{CONFIG['patience']})")
            if patience_counter >= CONFIG["patience"]:
                print(f"  🛑 Early stopping at epoch {epoch}")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final
    final_path = os.path.join(CONFIG["output_dir"], "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Final model saved to: {final_path}")

    # Save results
    results_path = os.path.join(CONFIG["output_dir"], "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("CORRECTION ENGINE — IMPROVED RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model       : {CONFIG['model_name']}\n")
        f.write(f"LR          : {CONFIG['learning_rate']}\n")
        f.write(f"Train Size  : {len(df_train)}\n")
        f.write(f"Best ROUGE-L: {best_rouge:.2f}\n\n")
        if rouge_scores:
            f.write("FINAL ROUGE SCORES:\n")
            for k, v in rouge_scores.items():
                f.write(f"  {k}: {v}\n")
    print(f"✅ Results saved to: {results_path}")

    # Loss plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs_done = range(1, len(train_losses)+1)
    axes[0].plot(epochs_done, train_losses, "b-o", label="Train Loss", lw=2)
    axes[0].plot(epochs_done, val_losses,   "r-o", label="Val Loss",   lw=2)
    axes[0].set_title("Correction Engine — Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_done, rouge_l_history, "g-o", label="ROUGE-L", lw=2)
    axes[1].axhline(y=30, color="orange", linestyle="--", label="Good threshold (30)")
    axes[1].set_title("ROUGE-L per Epoch", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_loss.png"), dpi=150)
    print(f"✅ Plot saved")

    # Test corrections
    print("\n--- Testing Correction Engine ---")
    test_cases = [
        ("Albert Einstein invented the telephone.",
         "Alexander Graham Bell is credited with inventing the telephone in 1876."),
        ("The Amazon is the longest river in the world.",
         "The Nile is the longest river at 6,650 km. The Amazon is the largest by water flow."),
        ("Shakespeare was born in London.",
         "William Shakespeare was born in Stratford-upon-Avon, Warwickshire, England in 1564."),
        ("Water boils at 50 degrees Celsius at sea level.",
         "Water boils at 100 degrees Celsius at standard atmospheric pressure."),
    ]

    print()
    for i, (wrong, ev) in enumerate(test_cases, 1):
        corrected = correct_statement(wrong, ev)
        print(f"  {i}. ❌ {wrong}")
        print(f"     ✅ {corrected}")
        print(f"     📖 {ev[:80]}...")
        print()

    print("=" * 60)
    print(f"  PHASE 6 COMPLETE | Best ROUGE-L: {best_rouge:.2f}")
    print("=" * 60)
