"""
=============================================================
  Phase 6 — Correction Engine Training
  Model  : google/flan-t5-base (fine-tuned text2text)
  GPU    : RTX 3050 Mobile 4GB — Linux
  Input  : outputs/correction_train.csv
  Output : corrector_model/best/
           corrector_model/final/
           corrector_model/results.txt
=============================================================
WHAT THIS DOES:
  Trains a text-to-text model that takes a hallucinated
  statement + evidence and generates the corrected version.

  Input  : "correct: [wrong statement] evidence: [evidence]"
  Output : "[corrected factual statement]"

  Example:
  Input  : "correct: Einstein invented the telephone.
            evidence: Alexander Graham Bell invented the
            telephone in 1876."
  Output : "The telephone was invented by Alexander Graham
            Bell in 1876."

WHY FLAN-T5?
  - Text-to-text model — perfect for correction task
  - flan-t5-base is 250MB — fits easily in 4GB VRAM
  - Already instruction-tuned — understands "correct:" prefix
  - Much smaller than GPT but good enough for correction

HOW TO RUN:
  Run AFTER Phase 3 and Phase 4 are complete
  python train_corrector.py
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
    T5Tokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

# For evaluating correction quality
# ROUGE measures overlap between generated and reference text
try:
    from evaluate import load as load_metric
    rouge = load_metric("rouge")
    ROUGE_AVAILABLE = True
except:
    ROUGE_AVAILABLE = False
    print("⚠ evaluate library not found — ROUGE scoring disabled")
    print("  Install with: pip install evaluate rouge-score")

# =============================================================
# CONFIGURATION
# =============================================================

CONFIG = {
    # Paths
    "train_path"      : "./outputs/correction_train.csv",
    "output_dir"      : "./corrector_model",

    # Model — flan-t5-base is the sweet spot for 4GB VRAM
    # flan-t5-small is faster but lower quality
    # flan-t5-large needs ~8GB VRAM — too big for our GPU
    "model_name"      : "google/flan-t5-small",

    # Input/output lengths
    # Input  = "correct: [statement] evidence: [evidence]"
    # Output = corrected statement
    "max_input_length"  : 256,   # longer inputs — contains both statement + evidence
    "max_output_length" : 128,   # output is just the corrected statement

    # Batch size — T5 is smaller than RoBERTa, 8 fits safely
    # Reduce to 4 if out-of-memory
    "batch_size"        : 1,

    # Gradient accumulation — effective batch = 8x4 = 32
    "grad_accum"        : 2,

    # Training
    "epochs"            : 3,
    "learning_rate"     : 3e-4,    # T5 uses higher LR than BERT models
    "warmup_ratio"      : 0.1,
    "weight_decay"      : 0.01,
    "max_grad_norm"     : 1.0,

    # Limit samples for first run
    # correction_train.csv has 20,000 rows
    # 10,000 is enough for a strong correction model
    # Set to None to use all 20,000
    "max_train_samples" : 10000,

    # Validation split — 10% of training data used for validation
    "val_split"         : 0.1,

    # Linux workers
    "num_workers"       : 4,

    "seed"              : 42,
}

# =============================================================
# STEP 1 — DEVICE SETUP
# =============================================================

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

print("=" * 60)
print("  Hallucination Detection — Phase 6 Correction Engine")
print("=" * 60)

if torch.cuda.is_available():
    device   = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n✅ GPU : {gpu_name}")
    print(f"   VRAM: {vram_gb:.1f} GB")
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
else:
    device = torch.device("cpu")
    print("\n⚠ GPU not found — running on CPU (slow)")

print(f"   Device: {device}\n")

# =============================================================
# STEP 2 — LOAD DATA
# =============================================================

print("--- Loading correction dataset ---")

df = pd.read_csv(CONFIG["train_path"], encoding="latin-1")

# Drop rows with missing values
df = df.dropna(subset=["wrong_statement", "correct_statement"]).reset_index(drop=True)

# Fill missing evidence with empty string
df["evidence"] = df["evidence"].fillna("").astype(str)
df["wrong_statement"]   = df["wrong_statement"].astype(str)
df["correct_statement"] = df["correct_statement"].astype(str)

# Remove rows where correct_statement is too short (likely noise)
df = df[df["correct_statement"].str.len() > 3].reset_index(drop=True)

# Limit samples if configured
if CONFIG["max_train_samples"] and len(df) > CONFIG["max_train_samples"]:
    df = df.sample(
        n            = CONFIG["max_train_samples"],
        random_state = CONFIG["seed"]
    ).reset_index(drop=True)
    print(f"   Samples capped at: {CONFIG['max_train_samples']}")

# Split into train and validation
val_size   = int(len(df) * CONFIG["val_split"])
train_size = len(df) - val_size

df_train = df.iloc[:train_size].reset_index(drop=True)
df_val   = df.iloc[train_size:].reset_index(drop=True)

print(f"   Train : {len(df_train)} samples")
print(f"   Val   : {len(df_val)} samples")
print(f"\n   Sample correction pair:")
print(f"     Wrong  : {df_train['wrong_statement'].iloc[0][:80]}...")
print(f"     Correct: {df_train['correct_statement'].iloc[0][:80]}...")
print()

# =============================================================
# STEP 3 — TOKENIZER
# T5 uses a different tokenizer than RoBERTa
# It is a SentencePiece tokenizer
# =============================================================

print("--- Loading T5 tokenizer ---")

# T5Tokenizer is specific to T5 family models
# Do NOT use AutoTokenizer here — it loads a fast tokenizer
# that has known issues with T5 padding
tokenizer = T5Tokenizer.from_pretrained(
    CONFIG["model_name"],
    legacy = False    # use new tokenizer behavior
)

print("   Tokenizer loaded\n")

# =============================================================
# STEP 4 — BUILD INPUT PROMPTS
# T5 is a text-to-text model — both input and output are text
# We use a prefix "correct:" to tell the model what task to do
# This is called instruction tuning / prompt engineering
# =============================================================

def build_input_prompt(wrong_statement, evidence):
    """
    Builds the input text for the T5 model.

    Format: "correct: [wrong statement] evidence: [evidence]"

    The "correct:" prefix tells Flan-T5 this is a correction task.
    The "evidence:" section gives the model factual context to work with.
    Without evidence, the model can only guess the correction.
    With evidence, it can generate a factually grounded correction.
    """
    if evidence and evidence.strip():
        return f"correct: {wrong_statement} evidence: {evidence}"
    else:
        # If no evidence available, still attempt correction
        return f"correct the following hallucinated statement: {wrong_statement}"


# =============================================================
# STEP 5 — DATASET CLASS
# T5 requires both input_ids AND labels (target token ids)
# =============================================================

class CorrectionDataset(Dataset):
    """
    Dataset for the correction task.
    Each sample contains:
      - input_ids      : tokenized "correct: [wrong] evidence: [ev]"
      - attention_mask : 1s for real tokens, 0s for padding
      - labels         : tokenized correct statement (target output)
    """

    def __init__(self, dataframe, tokenizer, max_input_len, max_output_len):
        self.data           = dataframe.reset_index(drop=True)
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Build input prompt
        input_text  = build_input_prompt(
            str(row["wrong_statement"]),
            str(row["evidence"])
        )
        # Target output — what the model should generate
        target_text = str(row["correct_statement"])

        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length  = self.max_input_len,
            truncation  = True,
            padding     = "max_length",
            return_tensors = "pt"
        )

        # Tokenize target output
        # We tokenize the target separately with max_output_length
        # Tokenize target output
        target_encoding = self.tokenizer(
           text_target    = target_text,
           max_length     = self.max_output_len,
           truncation     = True,
           padding        = "max_length",
           return_tensors = "pt"
)
        # Get label IDs
        labels = target_encoding["input_ids"].squeeze(0)

        # Replace padding token id with -100
        # T5's loss function ignores positions where label=-100
        # This means padding does not contribute to the loss
        # Without this, the model would try to predict padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids"      : input_encoding["input_ids"].squeeze(0),
            "attention_mask" : input_encoding["attention_mask"].squeeze(0),
            "labels"         : labels
        }


# Build datasets
train_dataset = CorrectionDataset(
    df_train, tokenizer,
    CONFIG["max_input_length"], CONFIG["max_output_length"]
)
val_dataset = CorrectionDataset(
    df_val, tokenizer,
    CONFIG["max_input_length"], CONFIG["max_output_length"]
)

# Build DataLoaders
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

print(f"--- DataLoaders ready ---")
print(f"   Train batches : {len(train_loader)}")
print(f"   Val batches   : {len(val_loader)}\n")

# =============================================================
# STEP 6 — LOAD MODEL
# T5ForConditionalGeneration is the encoder-decoder T5 model
# =============================================================

print("--- Loading Flan-T5-base model ---")
print("   (downloads ~250MB on first run, cached after)\n")

model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"])
model = model.to(device)

# Enable gradient checkpointing to save VRAM
model.gradient_checkpointing_enable()

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   Total parameters    : {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}\n")

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
# STEP 8 — TRAINING FUNCTION
# =============================================================

def train_one_epoch(model, loader, optimizer, scheduler, device, grad_accum, epoch):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):

        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)

        # T5 forward pass
        # decoder_input_ids are automatically created from labels
        # inside the model when labels are provided
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            labels         = labels
        )

        loss = outputs.loss / grad_accum
        loss.backward()

        total_loss += outputs.loss.item()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 100 == 0:
            avg_loss = total_loss / (step + 1)
            lr       = scheduler.get_last_lr()[0]
            print(f"   Epoch {epoch} | Step {step+1:>4}/{len(loader)} "
                  f"| Loss: {avg_loss:.4f} "
                  f"| LR: {lr:.2e}")

            if torch.cuda.is_available():
                used       = torch.cuda.memory_allocated() / 1e9
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"             GPU memory: {used:.2f}/{total_vram:.1f} GB")

    return total_loss / len(loader)


# =============================================================
# STEP 9 — VALIDATION FUNCTION
# Generates actual text outputs and computes ROUGE score
# =============================================================

def validate(model, loader, device, num_samples=100):
    """
    Generates corrections for validation samples and
    computes ROUGE score against the ground truth.

    ROUGE-1: unigram overlap (individual word matches)
    ROUGE-2: bigram overlap (two-word phrase matches)
    ROUGE-L: longest common subsequence
    """
    model.eval()
    total_loss    = 0
    generated     = []
    references    = []

    # Only generate text for first num_samples to save time
    generate_count = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels         = batch["labels"].to(device, non_blocking=True)

            # Compute loss
            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                labels         = labels
            )
            total_loss += outputs.loss.item()

            # Generate text for ROUGE evaluation
            if generate_count < num_samples:
                generated_ids = model.generate(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    max_length     = CONFIG["max_output_length"],
                    num_beams      = 4,       # beam search for better quality
                    early_stopping = True,
                    no_repeat_ngram_size = 2  # prevents repetitive outputs
                )

                # Decode generated token IDs back to text
                decoded_preds = tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens = True   # removes <pad>, </s> etc.
                )

                # Decode reference labels back to text
                # Replace -100 with pad_token_id before decoding
                label_ids = labels.clone()
                label_ids[label_ids == -100] = tokenizer.pad_token_id
                decoded_refs = tokenizer.batch_decode(
                    label_ids,
                    skip_special_tokens = True
                )

                generated.extend(decoded_preds)
                references.extend(decoded_refs)
                generate_count += len(input_ids)

    avg_loss = total_loss / len(loader)

    # Compute ROUGE scores
    rouge_scores = {}
    if ROUGE_AVAILABLE and generated:
        scores = rouge.compute(
            predictions = generated,
            references  = references
        )
        rouge_scores = {
            "rouge1": round(scores["rouge1"] * 100, 2),
            "rouge2": round(scores["rouge2"] * 100, 2),
            "rougeL": round(scores["rougeL"] * 100, 2),
        }

    return avg_loss, rouge_scores, generated[:5], references[:5]


# =============================================================
# STEP 10 — CORRECTION FUNCTION
# Call this after training to correct any hallucinated statement
# =============================================================

def correct_statement(wrong_statement, evidence="", num_beams=4):
    """
    Generates a corrected version of a hallucinated statement.

    Args:
        wrong_statement : the hallucinated text to correct
        evidence        : Wikipedia evidence to base correction on
        num_beams       : beam search width (higher = better but slower)

    Returns:
        dict with corrected text and metadata
    """
    model.eval()

    input_text = build_input_prompt(wrong_statement, evidence)

    # Tokenize input
    inputs = tokenizer(
        input_text,
        max_length     = CONFIG["max_input_length"],
        truncation     = True,
        padding        = "max_length",
        return_tensors = "pt"
    )

    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        # Generate corrected text using beam search
        output_ids = model.generate(
            input_ids            = input_ids,
            attention_mask       = attention_mask,
            max_length           = CONFIG["max_output_length"],
            num_beams            = num_beams,
            early_stopping       = True,
            no_repeat_ngram_size = 2,
            # Temperature controls randomness
            # 1.0 = standard, lower = more deterministic
            do_sample            = False
        )

    # Decode output token IDs back to text
    corrected = tokenizer.decode(
        output_ids[0],
        skip_special_tokens = True
    )

    return {
        "original"  : wrong_statement,
        "corrected" : corrected,
        "evidence"  : evidence[:200] + "..." if len(evidence) > 200 else evidence,
        "prompt"    : input_text[:200]
    }


# =============================================================
# STEP 11 — MAIN TRAINING LOOP
# =============================================================

if __name__ == "__main__":

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("\n" + "=" * 60)
    print("  CORRECTION ENGINE TRAINING STARTED")
    print("=" * 60)

    best_rouge   = 0
    train_losses = []
    val_losses   = []

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch} / {CONFIG['epochs']}")
        print(f"{'='*60}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, CONFIG["grad_accum"], epoch
        )
        train_losses.append(train_loss)

        # Validate
        val_loss, rouge_scores, sample_preds, sample_refs = validate(
            model, val_loader, device
        )
        val_losses.append(val_loss)

        print(f"\n  Results — Epoch {epoch}:")
        print(f"    Train Loss : {train_loss:.4f}")
        print(f"    Val Loss   : {val_loss:.4f}")

        if rouge_scores:
            print(f"    ROUGE-1    : {rouge_scores.get('rouge1', 'N/A')}")
            print(f"    ROUGE-2    : {rouge_scores.get('rouge2', 'N/A')}")
            print(f"    ROUGE-L    : {rouge_scores.get('rougeL', 'N/A')}")
            current_rouge = rouge_scores.get("rougeL", 0)
        else:
            current_rouge = -val_loss   # use negative loss if ROUGE unavailable

        # Show sample predictions
        print(f"\n  Sample corrections (Epoch {epoch}):")
        for pred, ref in zip(sample_preds[:2], sample_refs[:2]):
            print(f"    Generated : {pred[:100]}")
            print(f"    Reference : {ref[:100]}")
            print()

        # Save best model
        if current_rouge > best_rouge:
            best_rouge = current_rouge
            best_path  = os.path.join(CONFIG["output_dir"], "best")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"  ✅ Best model saved")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    final_path = os.path.join(CONFIG["output_dir"], "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ Final model saved to: {final_path}")

    # ── SAVE RESULTS ──────────────────────────────────────
    results_path = os.path.join(CONFIG["output_dir"], "results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("CORRECTION ENGINE — RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model       : {CONFIG['model_name']}\n")
        f.write(f"Max Input   : {CONFIG['max_input_length']}\n")
        f.write(f"Max Output  : {CONFIG['max_output_length']}\n")
        f.write(f"Batch Size  : {CONFIG['batch_size']} (effective: {CONFIG['batch_size']*CONFIG['grad_accum']})\n")
        f.write(f"Epochs      : {CONFIG['epochs']}\n")
        f.write(f"Train Size  : {len(df_train)}\n")
        f.write(f"Val Size    : {len(df_val)}\n\n")
        if rouge_scores:
            f.write("FINAL ROUGE SCORES:\n")
            f.write(f"  ROUGE-1 : {rouge_scores.get('rouge1')}\n")
            f.write(f"  ROUGE-2 : {rouge_scores.get('rouge2')}\n")
            f.write(f"  ROUGE-L : {rouge_scores.get('rougeL')}\n")
    print(f"✅ Results saved to: {results_path}")

    # ── LOSS PLOT ──────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, CONFIG["epochs"] + 1)
    plt.plot(epochs_range, train_losses, "b-o", label="Train Loss", linewidth=2)
    plt.plot(epochs_range, val_losses,   "r-o", label="Val Loss",   linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Correction Engine — Training & Validation Loss", fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(CONFIG["output_dir"], "training_loss.png")
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Plot saved to: {plot_path}")

    # ── TEST CORRECTIONS ──────────────────────────────────
    print("\n--- Testing Correction Engine ---")

    test_cases = [
        {
            "wrong"    : "Albert Einstein invented the telephone.",
            "evidence" : "Alexander Graham Bell is credited with inventing the telephone in 1876."
        },
        {
            "wrong"    : "The Great Wall of China is visible from space with the naked eye.",
            "evidence" : "Contrary to popular belief, the Great Wall of China is not visible from space with the naked eye."
        },
        {
            "wrong"    : "The Amazon is the longest river in the world.",
            "evidence" : "The Nile is generally considered the longest river in the world at 6,650 km. The Amazon is the largest by water flow."
        },
        {
            "wrong"    : "Shakespeare was born in London.",
            "evidence" : "William Shakespeare was born in Stratford-upon-Avon, Warwickshire, England in 1564."
        },
        {
            "wrong"    : "Water boils at 50 degrees Celsius at sea level.",
            "evidence" : "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure (sea level)."
        },
    ]

    print()
    for i, case in enumerate(test_cases, 1):
        result = correct_statement(case["wrong"], case["evidence"])
        print(f"  Example {i}:")
        print(f"    ❌ Wrong    : {result['original']}")
        print(f"    ✅ Corrected: {result['corrected']}")
        print(f"    📖 Evidence : {case['evidence'][:100]}...")
        print()

    print("=" * 60)
    print("  PHASE 6 COMPLETE")
    print(f"  Best model : {os.path.abspath(os.path.join(CONFIG['output_dir'], 'best'))}")
    print(f"  Final model: {os.path.abspath(final_path)}")
    print("=" * 60)
