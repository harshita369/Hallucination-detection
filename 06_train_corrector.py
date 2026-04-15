"""
=============================================================
  Phase 6 — Correction Engine
  Model  : google/flan-t5-small
  GPU    : RTX 3050 Mobile 4GB — Linux

  FIXES:
  1. Prompt now demands a full complete sentence explicitly
  2. min_length=15, length_penalty=2.0 — forces full sentences
  3. Lower LR (5e-5) for stable training
  4. All 20,000 correction samples used
  5. Early stopping on ROUGE-L
  6. no_repeat_ngram_size=3 prevents repetition
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
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW

try:
    from evaluate import load as load_metric
    rouge_metric = load_metric("rouge")
    ROUGE_OK = True
except:
    ROUGE_OK = False
    print("⚠ pip install evaluate rouge-score")

CONFIG = {
    "train_path"       : "./outputs/correction_train.csv",
    "output_dir"       : "./corrector_model",
    "model_name"       : "google/flan-t5-small",
    "max_input_length" : 256,
    "max_output_length": 128,
    "batch_size"       : 8,
    "grad_accum"       : 4,
    "epochs"           : 6,
    "learning_rate"    : 5e-5,    # lower than before for stable training
    "warmup_ratio"     : 0.1,
    "weight_decay"     : 0.01,
    "max_grad_norm"    : 1.0,
    "patience"         : 2,
    "max_train_samples": None,    # use all 20,000
    "val_split"        : 0.1,
    "use_fp16"         : True,
    "num_workers"      : 4,
    "seed"             : 42,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
torch.manual_seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"])

print("=" * 60)
print("  Phase 6 — Correction Engine")
print("=" * 60)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n✅ GPU : {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
else:
    device = torch.device("cpu")
    CONFIG["use_fp16"] = False
print(f"   Device: {device}\n")

# ── Load data ─────────────────────────────────────────────
print("--- Loading correction dataset ---")
df = pd.read_csv(CONFIG["train_path"], encoding="latin-1")
df = df.dropna(subset=["wrong_statement","correct_statement"]).reset_index(drop=True)
df["evidence"]          = df["evidence"].fillna("").astype(str)
df["wrong_statement"]   = df["wrong_statement"].astype(str)
df["correct_statement"] = df["correct_statement"].astype(str)
df = df[df["correct_statement"].str.len() > 5].reset_index(drop=True)
df = df[df["wrong_statement"].str.len()   > 5].reset_index(drop=True)

if CONFIG["max_train_samples"] and len(df) > CONFIG["max_train_samples"]:
    df = df.sample(n=CONFIG["max_train_samples"], random_state=CONFIG["seed"]).reset_index(drop=True)

val_size = int(len(df) * CONFIG["val_split"])
df_train = df.iloc[:len(df)-val_size].reset_index(drop=True)
df_val   = df.iloc[len(df)-val_size:].reset_index(drop=True)
print(f"   Train: {len(df_train)} | Val: {len(df_val)}\n")

# ── Tokenizer ─────────────────────────────────────────────
tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
print("   Tokenizer loaded\n")

# ── Prompt builder — explicit full-sentence instruction ───
def build_prompt(wrong, evidence):
    """
    Improved prompt that forces the model to output
    a complete, grammatically correct sentence.
    """
    if evidence and evidence.strip() and evidence.strip() != "nan":
        return (
            f"Task: Correct the factual error in the statement below. "
            f"Use the evidence provided. "
            f"Write one complete corrected sentence.\n"
            f"Evidence: {evidence.strip()}\n"
            f"Incorrect statement: {wrong}\n"
            f"Corrected statement:"
        )
    else:
        return (
            f"Task: Correct the factual error in the statement below. "
            f"Write one complete corrected sentence.\n"
            f"Incorrect statement: {wrong}\n"
            f"Corrected statement:"
        )

# ── Dataset ───────────────────────────────────────────────
class CorrectionDataset(Dataset):
    def __init__(self, df, tokenizer, max_in, max_out):
        self.data = df.reset_index(drop=True)
        self.tok  = tokenizer
        self.max_in  = max_in
        self.max_out = max_out

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inp = build_prompt(str(row["wrong_statement"]), str(row["evidence"]))
        tgt = str(row["correct_statement"])

        in_enc = self.tok(inp, max_length=self.max_in, truncation=True,
                          padding="max_length", return_tensors="pt")
        tg_enc = self.tok(text_target=tgt, max_length=self.max_out,
                          truncation=True, padding="max_length", return_tensors="pt")

        labels = tg_enc["input_ids"].squeeze(0)
        labels[labels == self.tok.pad_token_id] = -100

        return {
            "input_ids"      : in_enc["input_ids"].squeeze(0),
            "attention_mask" : in_enc["attention_mask"].squeeze(0),
            "labels"         : labels
        }

train_ds = CorrectionDataset(df_train, tokenizer, CONFIG["max_input_length"],
                              CONFIG["max_output_length"])
val_ds   = CorrectionDataset(df_val,   tokenizer, CONFIG["max_input_length"],
                              CONFIG["max_output_length"])

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                          num_workers=CONFIG["num_workers"], pin_memory=True,
                          persistent_workers=True)
val_loader   = DataLoader(val_ds, batch_size=CONFIG["batch_size"]*2, shuffle=False,
                          num_workers=CONFIG["num_workers"], pin_memory=True,
                          persistent_workers=True)

print(f"   Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

# ── Model ─────────────────────────────────────────────────
print("--- Loading Flan-T5-small ---")
model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"]).to(device)
model.gradient_checkpointing_enable()
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                  weight_decay=CONFIG["weight_decay"])
total_steps  = (len(train_loader) // CONFIG["grad_accum"]) * CONFIG["epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler       = GradScaler(enabled=CONFIG["use_fp16"])

print(f"   Epochs: {CONFIG['epochs']} | LR: {CONFIG['learning_rate']} "
      f"| Effective batch: {CONFIG['batch_size']*CONFIG['grad_accum']}\n")

# ── Train ─────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler, device, grad_accum, ep):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(loader):
        ids   = batch["input_ids"].to(device, non_blocking=True)
        mask  = batch["attention_mask"].to(device, non_blocking=True)
        lbls  = batch["labels"].to(device, non_blocking=True)
        with autocast(enabled=CONFIG["use_fp16"]):
            out  = model(input_ids=ids, attention_mask=mask, labels=lbls)
            loss = out.loss / grad_accum
        scaler.scale(loss).backward()
        total_loss += out.loss.item()
        if (step+1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            scaler.step(optimizer); scaler.update(); scheduler.step()
            optimizer.zero_grad()
        if (step+1) % 100 == 0:
            print(f"   Ep {ep} | Step {step+1:>4}/{len(loader)} "
                  f"| Loss: {total_loss/(step+1):.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e}")
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated()/1e9
                tot  = torch.cuda.get_device_properties(0).total_memory/1e9
                print(f"             GPU: {used:.2f}/{tot:.1f} GB")
    return total_loss / len(loader)

def validate(model, loader, device, num_gen=200):
    model.eval()
    total_loss = 0
    generated, references = [], []
    gen_count = 0
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(device, non_blocking=True)
            mask  = batch["attention_mask"].to(device, non_blocking=True)
            lbls  = batch["labels"].to(device, non_blocking=True)
            with autocast(enabled=CONFIG["use_fp16"]):
                out = model(input_ids=ids, attention_mask=mask, labels=lbls)
            total_loss += out.loss.item()
            if gen_count < num_gen:
                gen_ids = model.generate(
                    input_ids=ids, attention_mask=mask,
                    max_length           = CONFIG["max_output_length"],
                    min_length           = 15,
                    num_beams            = 4,
                    length_penalty       = 2.0,
                    early_stopping       = True,
                    no_repeat_ngram_size = 3
                )
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                lab   = lbls.clone()
                lab[lab == -100] = tokenizer.pad_token_id
                refs  = tokenizer.batch_decode(lab, skip_special_tokens=True)
                generated.extend(preds); references.extend(refs)
                gen_count += len(ids)
    avg_loss = total_loss / len(loader)
    scores = {}
    if ROUGE_OK and generated:
        r = rouge_metric.compute(predictions=generated, references=references)
        scores = {"rouge1": round(r["rouge1"]*100,2),
                  "rouge2": round(r["rouge2"]*100,2),
                  "rougeL": round(r["rougeL"]*100,2)}
    return avg_loss, scores, generated[:3], references[:3]

def correct_statement(wrong, evidence=""):
    """Generate corrected statement — used by predict.py and evaluate.py"""
    model.eval()
    prompt = build_prompt(wrong, evidence)
    enc = tokenizer(prompt, max_length=CONFIG["max_input_length"],
                    truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids            = enc["input_ids"].to(device),
            attention_mask       = enc["attention_mask"].to(device),
            max_length           = CONFIG["max_output_length"],
            min_length           = 15,
            num_beams            = 4,
            length_penalty       = 2.0,
            early_stopping       = True,
            no_repeat_ngram_size = 3,
            do_sample            = False
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)

# ── Main loop ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CORRECTION ENGINE TRAINING STARTED")
    print("=" * 60)

    best_rouge, patience_counter = 0, 0
    train_losses, val_losses, rouge_hist = [], [], []

    for epoch in range(1, CONFIG["epochs"]+1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch} / {CONFIG['epochs']}")
        print(f"{'='*60}")

        tr_loss = train_epoch(model, train_loader, optimizer, scheduler,
                              scaler, device, CONFIG["grad_accum"], epoch)
        train_losses.append(tr_loss)

        val_loss, scores, sample_preds, sample_refs = validate(model, val_loader, device)
        val_losses.append(val_loss)
        rouge_l = scores.get("rougeL", 0)
        rouge_hist.append(rouge_l)

        print(f"\n  Epoch {epoch} Results:")
        print(f"    Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")
        if scores:
            print(f"    ROUGE-1: {scores['rouge1']} | ROUGE-2: {scores['rouge2']} | ROUGE-L: {scores['rougeL']}")

        print(f"\n  Sample corrections:")
        for p, r in zip(sample_preds, sample_refs):
            print(f"    Generated : {p[:100]}")
            print(f"    Reference : {r[:100]}\n")

        if rouge_l > best_rouge:
            best_rouge, patience_counter = rouge_l, 0
            bp = os.path.join(CONFIG["output_dir"], "best")
            model.save_pretrained(bp); tokenizer.save_pretrained(bp)
            print(f"  ✅ Best model saved — ROUGE-L: {best_rouge:.2f}")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement ({patience_counter}/{CONFIG['patience']})")
            if patience_counter >= CONFIG["patience"]:
                print("  🛑 Early stopping"); break

        if torch.cuda.is_available(): torch.cuda.empty_cache()

    fp = os.path.join(CONFIG["output_dir"], "final")
    model.save_pretrained(fp); tokenizer.save_pretrained(fp)
    print(f"\n✅ Final model saved: {fp}")

    # Save results
    with open(os.path.join(CONFIG["output_dir"], "results.txt"), "w") as f:
        f.write("CORRECTION ENGINE — RESULTS\n" + "="*50 + "\n\n")
        f.write(f"Model        : {CONFIG['model_name']}\n")
        f.write(f"LR           : {CONFIG['learning_rate']}\n")
        f.write(f"Train Size   : {len(df_train)}\n")
        f.write(f"Best ROUGE-L : {best_rouge:.2f}\n\n")
        if scores:
            f.write(f"ROUGE-1: {scores.get('rouge1')}\n")
            f.write(f"ROUGE-2: {scores.get('rouge2')}\n")
            f.write(f"ROUGE-L: {scores.get('rougeL')}\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ep_done = range(1, len(train_losses)+1)
    axes[0].plot(ep_done, train_losses, "b-o", label="Train Loss", lw=2)
    axes[0].plot(ep_done, val_losses,   "r-o", label="Val Loss",   lw=2)
    axes[0].set_title("Loss Curves", fontweight="bold")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(ep_done, rouge_hist, "g-o", label="ROUGE-L", lw=2)
    axes[1].axhline(y=30, color="orange", linestyle="--", label="Good (30)")
    axes[1].set_title("ROUGE-L per Epoch", fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "training_loss.png"), dpi=150)
    print("✅ Plot saved")

    # Test corrections — should now produce full sentences
    print("\n--- Testing Correction Engine ---")
    test_cases = [
        ("Albert Einstein invented the telephone.",
         "Alexander Graham Bell is credited with inventing the telephone in 1876."),
        ("The Amazon is the longest river in the world.",
         "The Nile is the longest river at 6,650 km. The Amazon is the largest by water flow."),
        ("Shakespeare was born in London.",
         "William Shakespeare was born in Stratford-upon-Avon, England in 1564."),
        ("Water boils at 50 degrees Celsius at sea level.",
         "Water boils at 100 degrees Celsius at standard atmospheric pressure."),
        ("The Great Wall of China is visible from space.",
         "The Great Wall of China is NOT visible from space with the naked eye."),
    ]
    print()
    for i, (wrong, ev) in enumerate(test_cases, 1):
        corrected = correct_statement(wrong, ev)
        print(f"  {i}. ❌ {wrong}")
        print(f"     ✅ {corrected}")
        print(f"     📖 {ev[:80]}...")
        print()

    print("="*60)
    print(f"  PHASE 6 COMPLETE | Best ROUGE-L: {best_rouge:.2f}")
    print("="*60)
