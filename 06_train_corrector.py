"""
=============================================================
  Phase 6 — Correction Engine (CPU Training)
  Model  : google/flan-t5-small
  Device : CPU  ← intentional, see WHY below

  WHY CPU AND NOT GPU:
  ────────────────────────────────────────────────────────
  When Python imports retriever.py at the top level,
  it loads sentence-transformers + Wikipedia API.
  When evaluate.py is imported, it loads 3 models onto GPU:
    - RoBERTa detector  (~500MB VRAM)
    - NLI-RoBERTa       (~500MB VRAM)
    - Flan-T5-small     (~300MB VRAM)
  This leaves only ~1.4GB free — not enough to train T5 FP32.

  THE FIX: Train T5 on CPU.
  flan-t5-small is only 77M params. On CPU it is slower
  (~15-20 min per epoch) but trains correctly with no OOM.
  Quality is identical — device doesn't affect model weights.

  HOW TO RUN:
  ── Run this in a SEPARATE terminal from evaluate.py ────
  python 06_train_corrector.py
  ── Expected time: ~15-20 min per epoch on CPU ──────────
=============================================================
"""

import os
import gc
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

try:
    from evaluate import load as load_metric
    rouge_metric = load_metric("rouge")
    ROUGE_OK = True
except:
    ROUGE_OK = False
    print("⚠ pip install evaluate rouge-score")

# =============================================================
# CONFIGURATION
# =============================================================
CONFIG = {
    "train_path"       : "./outputs/correction_train.csv",
    "output_dir"       : "./corrector_model",
    "model_name"       : "google/flan-t5-small",

    "max_input_length" : 128,
    "max_output_length": 64,

    # CPU batch — larger is fine on CPU (no VRAM limit)
    "batch_size"       : 4,
    "grad_accum"       : 8,       # effective batch = 32

    "epochs"           : 3,       # 3 epochs on CPU is reasonable
    "learning_rate"    : 3e-5,
    "warmup_ratio"     : 0.05,
    "weight_decay"     : 0.01,
    "max_grad_norm"    : 0.5,
    "patience"         : 2,
    "max_train_samples": 10000,   # 10k samples keeps training < 2 hours on CPU
    "val_split"        : 0.1,
    "use_fp16"         : False,   # FP16 not supported on CPU

    "num_workers"      : 0,       # 0 workers for CPU training (avoids fork issues)
    "seed"             : 42,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# Force CPU — avoids all VRAM contention with other loaded models
device = torch.device("cpu")

print("=" * 60)
print("  Phase 6 — Correction Engine")
print("=" * 60)
print(f"\n   Device : CPU (intentional — avoids VRAM conflict)")
print(f"   Model  : {CONFIG['model_name']} (77M params)")
print(f"   Reason : GPU held by retriever + evaluate imports")
print(f"   Samples: {CONFIG['max_train_samples']} training samples")
print(f"   Time   : ~15-20 min per epoch on CPU\n")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    free_vram = (torch.cuda.get_device_properties(0).total_memory
                 - torch.cuda.memory_allocated()) / 1e9
    print(f"   GPU available ({gpu_name}) but intentionally NOT used")
    print(f"   GPU free VRAM: {free_vram:.2f} GB (occupied by other models)\n")

# =============================================================
# LOAD DATA
# =============================================================
print("--- Loading dataset ---")
df = pd.read_csv(CONFIG["train_path"], encoding="latin-1")
df = df.dropna(subset=["wrong_statement", "correct_statement"]).reset_index(drop=True)
df["evidence"]          = df["evidence"].fillna("").astype(str)
df["wrong_statement"]   = df["wrong_statement"].astype(str)
df["correct_statement"] = df["correct_statement"].astype(str)
df = df[df["correct_statement"].str.len() > 5].reset_index(drop=True)
df = df[df["wrong_statement"].str.len()   > 5].reset_index(drop=True)

if CONFIG["max_train_samples"] and len(df) > CONFIG["max_train_samples"]:
    df = df.sample(n=CONFIG["max_train_samples"],
                   random_state=CONFIG["seed"]).reset_index(drop=True)

val_size = int(len(df) * CONFIG["val_split"])
df_train = df.iloc[:len(df) - val_size].reset_index(drop=True)
df_val   = df.iloc[len(df) - val_size:].reset_index(drop=True)
print(f"   Train: {len(df_train)} | Val: {len(df_val)}\n")

# =============================================================
# TOKENIZER
# =============================================================
print("--- Loading tokenizer ---")
tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
print("   Tokenizer loaded\n")

# =============================================================
# PROMPT — short and clean
# =============================================================
def build_prompt(wrong, evidence):
    if evidence and evidence.strip() and evidence.strip() != "nan":
        ev = evidence.strip()[:100]
        return f"Fix this incorrect statement. Evidence: {ev} Statement: {wrong} Correction:"
    return f"Fix this incorrect statement. Statement: {wrong} Correction:"

# =============================================================
# DATASET
# =============================================================
class CorrectionDataset(Dataset):
    def __init__(self, df, tok, max_in, max_out):
        self.data    = df.reset_index(drop=True)
        self.tok     = tok
        self.max_in  = max_in
        self.max_out = max_out

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inp = build_prompt(str(row["wrong_statement"]), str(row["evidence"]))
        tgt = str(row["correct_statement"])[:180]

        in_enc = self.tok(
            inp, max_length=self.max_in,
            truncation=True, padding="max_length",
            return_tensors="pt"
        )
        tg_enc = self.tok(
            text_target=tgt, max_length=self.max_out,
            truncation=True, padding="max_length",
            return_tensors="pt"
        )
        labels = tg_enc["input_ids"].squeeze(0)
        labels[labels == self.tok.pad_token_id] = -100

        return {
            "input_ids"     : in_enc["input_ids"].squeeze(0),
            "attention_mask": in_enc["attention_mask"].squeeze(0),
            "labels"        : labels
        }

train_ds = CorrectionDataset(df_train, tokenizer,
                              CONFIG["max_input_length"],
                              CONFIG["max_output_length"])
val_ds   = CorrectionDataset(df_val, tokenizer,
                              CONFIG["max_input_length"],
                              CONFIG["max_output_length"])

train_loader = DataLoader(
    train_ds,
    batch_size  = CONFIG["batch_size"],
    shuffle     = True,
    num_workers = CONFIG["num_workers"],
    pin_memory  = False   # pin_memory=False on CPU
)
val_loader = DataLoader(
    val_ds,
    batch_size  = CONFIG["batch_size"] * 2,
    shuffle     = False,
    num_workers = CONFIG["num_workers"],
    pin_memory  = False
)

print(f"   Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

# =============================================================
# MODEL — loaded directly on CPU
# =============================================================
print("--- Loading Flan-T5-small on CPU ---")
model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"])
model = model.to(device)   # CPU
# No gradient checkpointing needed on CPU — memory is not a constraint
print(f"   Parameters : {sum(p.numel() for p in model.parameters()):,}")
print(f"   Device     : {next(model.parameters()).device}\n")

# =============================================================
# OPTIMIZER + SCHEDULER
# =============================================================
optimizer    = AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                     weight_decay=CONFIG["weight_decay"], eps=1e-8)
total_steps  = (len(train_loader) // CONFIG["grad_accum"]) * CONFIG["epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f"--- Training config ---")
print(f"   Epochs      : {CONFIG['epochs']} (patience={CONFIG['patience']})")
print(f"   Eff. batch  : {CONFIG['batch_size'] * CONFIG['grad_accum']}")
print(f"   LR          : {CONFIG['learning_rate']}")
print(f"   Input len   : {CONFIG['max_input_length']} | "
      f"Output len: {CONFIG['max_output_length']}\n")

# =============================================================
# TRAINING FUNCTION
# =============================================================
def train_epoch(ep):
    model.train()
    total_loss  = 0
    valid_steps = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        # No .to(device) needed — already on CPU
        ids   = batch["input_ids"]
        mask  = batch["attention_mask"]
        lbls  = batch["labels"]

        out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
        loss = out.loss

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        (loss / CONFIG["grad_accum"]).backward()
        total_loss  += loss.item()
        valid_steps += 1

        if (step + 1) % CONFIG["grad_accum"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           CONFIG["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 100 == 0 and valid_steps > 0:
            avg = total_loss / valid_steps
            lr  = scheduler.get_last_lr()[0]
            pct = (step + 1) / len(train_loader) * 100
            print(f"   Ep {ep} | Step {step+1:>4}/{len(train_loader)} "
                  f"({pct:.0f}%) | Loss: {avg:.4f} | LR: {lr:.2e}")

    return total_loss / valid_steps if valid_steps > 0 else float("nan")

# =============================================================
# VALIDATION FUNCTION
# =============================================================
def validate(num_gen=80):
    model.eval()
    total_loss = 0
    valid      = 0
    generated  = []
    references = []
    gen_count  = 0

    with torch.no_grad():
        for batch in val_loader:
            ids   = batch["input_ids"]
            mask  = batch["attention_mask"]
            lbls  = batch["labels"]

            out = model(input_ids=ids, attention_mask=mask, labels=lbls)
            if not (torch.isnan(out.loss) or torch.isinf(out.loss)):
                total_loss += out.loss.item()
                valid      += 1

            if gen_count < num_gen:
                gen_ids = model.generate(
                    input_ids              = ids,
                    attention_mask         = mask,
                    max_length             = CONFIG["max_output_length"],
                    min_length             = 5,
                    num_beams              = 2,
                    early_stopping         = True,
                    no_repeat_ngram_size   = 2
                )
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                lab   = lbls.clone()
                lab[lab == -100] = tokenizer.pad_token_id
                refs  = tokenizer.batch_decode(lab, skip_special_tokens=True)
                generated.extend(preds)
                references.extend(refs)
                gen_count += len(ids)

    avg_loss = total_loss / valid if valid > 0 else float("nan")
    scores   = {}
    if ROUGE_OK and generated:
        r = rouge_metric.compute(predictions=generated, references=references)
        scores = {
            "rouge1": round(r["rouge1"] * 100, 2),
            "rouge2": round(r["rouge2"] * 100, 2),
            "rougeL": round(r["rougeL"] * 100, 2),
        }
    return avg_loss, scores, generated[:2], references[:2]

# =============================================================
# CORRECTION FUNCTION
# =============================================================
def correct(wrong, evidence=""):
    model.eval()
    prompt = build_prompt(wrong, evidence)
    enc    = tokenizer(prompt, max_length=CONFIG["max_input_length"],
                       truncation=True, padding="max_length",
                       return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids            = enc["input_ids"],
            attention_mask       = enc["attention_mask"],
            max_length           = CONFIG["max_output_length"],
            min_length           = 5,
            num_beams            = 2,
            early_stopping       = True,
            no_repeat_ngram_size = 2,
            do_sample            = False
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)

# =============================================================
# MAIN TRAINING LOOP
# =============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("  TRAINING STARTED (CPU — no VRAM conflicts)")
    print("=" * 60)

    best_rouge, patience_cnt = 0, 0
    tr_losses, vl_losses, rouge_hist = [], [], []

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n{'='*60}\n  Epoch {epoch}/{CONFIG['epochs']}\n{'='*60}")

        tr_loss = train_epoch(epoch)
        tr_losses.append(tr_loss)

        vl_loss, scores, sample_preds, sample_refs = validate()
        vl_losses.append(vl_loss)
        rouge_l = scores.get("rougeL", 0)
        rouge_hist.append(rouge_l)

        print(f"\n  Epoch {epoch}: Train={tr_loss:.4f} | Val={vl_loss:.4f}")
        if scores:
            print(f"  ROUGE-1:{scores['rouge1']} | "
                  f"ROUGE-2:{scores['rouge2']} | "
                  f"ROUGE-L:{scores['rougeL']}")

        print("\n  Sample corrections:")
        for p, r in zip(sample_preds, sample_refs):
            print(f"    Generated : {p[:100]}")
            print(f"    Reference : {r[:100]}\n")

        if rouge_l > best_rouge:
            best_rouge, patience_cnt = rouge_l, 0
            bp = os.path.join(CONFIG["output_dir"], "best")
            model.save_pretrained(bp)
            tokenizer.save_pretrained(bp)
            print(f"  ✅ Best model saved — ROUGE-L: {best_rouge:.2f}")
        else:
            patience_cnt += 1
            print(f"  ⚠ No improvement ({patience_cnt}/{CONFIG['patience']})")
            if patience_cnt >= CONFIG["patience"]:
                print("  🛑 Early stopping")
                break

    # Save final model
    fp = os.path.join(CONFIG["output_dir"], "final")
    model.save_pretrained(fp)
    tokenizer.save_pretrained(fp)
    print(f"\n✅ Final model saved: {fp}")

    # Save results
    with open(os.path.join(CONFIG["output_dir"], "results.txt"), "w") as f:
        f.write("CORRECTION ENGINE RESULTS\n" + "=" * 50 + "\n\n")
        f.write(f"Model        : {CONFIG['model_name']}\n")
        f.write(f"Device       : CPU\n")
        f.write(f"LR           : {CONFIG['learning_rate']}\n")
        f.write(f"Train size   : {len(df_train)}\n")
        f.write(f"Best ROUGE-L : {best_rouge:.2f}\n")
        if scores:
            f.write(f"\nFINAL ROUGE SCORES:\n")
            f.write(f"  ROUGE-1 : {scores.get('rouge1')}\n")
            f.write(f"  ROUGE-2 : {scores.get('rouge2')}\n")
            f.write(f"  ROUGE-L : {scores.get('rougeL')}\n")
    print("✅ Results saved")

    # Loss plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ep_d = range(1, len(tr_losses) + 1)
    axes[0].plot(ep_d, tr_losses, "b-o", label="Train", lw=2)
    axes[0].plot(ep_d, vl_losses, "r-o", label="Val",   lw=2)
    axes[0].set_title("Loss Curves", fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep_d, rouge_hist, "g-o", label="ROUGE-L", lw=2)
    axes[1].axhline(y=30, color="orange", linestyle="--", label="Good (30)")
    axes[1].set_title("ROUGE-L per Epoch", fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_loss.png"), dpi=150)
    print("✅ Plot saved")

    # Test corrections
    print("\n--- Testing Correction Engine ---\n")
    test_cases = [
        ("Albert Einstein invented the telephone.",
         "Alexander Graham Bell invented the telephone in 1876."),
        ("The Amazon is the longest river in the world.",
         "The Nile is the longest river at 6,650 km."),
        ("Shakespeare was born in London.",
         "Shakespeare was born in Stratford-upon-Avon in 1564."),
        ("Water boils at 50 degrees Celsius.",
         "Water boils at 100 degrees Celsius at sea level."),
        ("1 liter is equal to 1200 ml.",
         "1 liter is equal to 1000 milliliters."),
        ("Domino's is a sushi restaurant.",
         "Domino's Pizza is an American pizza restaurant chain."),
    ]
    for i, (wrong, ev) in enumerate(test_cases, 1):
        corrected = correct(wrong, ev)
        print(f"  {i}. ❌ {wrong}")
        print(f"     ✅ {corrected}\n")

    print("=" * 60)
    print(f"  PHASE 6 COMPLETE | Best ROUGE-L: {best_rouge:.2f}")
    print(f"  Model: {os.path.abspath(os.path.join(CONFIG['output_dir'], 'best'))}")
    print("=" * 60)
