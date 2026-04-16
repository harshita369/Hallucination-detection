"""
=============================================================
  Phase 6 — Correction Engine v3
  Model  : google/flan-t5-small
  GPU    : RTX 3050 Mobile 4GB — Linux

  FIXES FOR NaN LOSS:
  ──────────────────────────────────────────────────────
  Problem: Loss = NaN throughout all epochs
  Root cause: FP16 (mixed precision) causes gradient
    overflow with T5 + cosine scheduler + long prompts.
    The new longer prompt increased sequence length and
    with FP16, some attention computations overflow to inf
    which then becomes NaN in the loss.

  FIX 1: Disable FP16 for T5 (use full float32)
    T5 is smaller (77M params) so it fits fine in fp32
    on 4GB VRAM.

  FIX 2: Use linear scheduler (cosine was too aggressive)

  FIX 3: Very low LR (3e-5) with short warmup

  FIX 4: NaN detection — skip bad batches automatically

  FIX 5: Gradient clipping at 0.5 (tighter than before)

  CORRECTIONS:
  - min_length=10 forces full sentence output
  - length_penalty=1.5 rewards longer outputs
  - Explicit sentence-completion prompt
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
    T5Tokenizer, T5ForConditionalGeneration,
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

CONFIG = {
    "train_path"       : "./outputs/correction_train.csv",
    "output_dir"       : "./corrector_model",
    "model_name"       : "google/flan-t5-small",
    "max_input_length" : 256,
    "max_output_length": 128,
    "batch_size"       : 4,
    "grad_accum"       : 8,      # effective batch = 32
    "epochs"           : 5,
    "learning_rate"    : 3e-5,   # lower LR
    "warmup_ratio"     : 0.05,   # short warmup
    "weight_decay"     : 0.01,
    "max_grad_norm"    : 0.5,    # tighter clipping
    "patience"         : 2,
    "max_train_samples": None,
    "val_split"        : 0.1,
    # FIX 1: NO FP16 for T5 — causes NaN
    "use_fp16"         : False,
    "num_workers"      : 2,
    "seed"             : 42,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
torch.manual_seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"])

print("="*60)
print("  Phase 6 — Correction Engine v3 (NaN Fixed)")
print("="*60)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n✅ GPU : {torch.cuda.get_device_name(0)}")
    print(f"   FP16: DISABLED for T5 (prevents NaN loss)")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
print(f"   Device: {device}\n")

# ── Load data ─────────────────────────────────────────────
print("--- Loading dataset ---")
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

# ── Prompt — shorter to avoid overflow ───────────────────
def build_prompt(wrong, evidence):
    """
    Shorter prompt to reduce risk of NaN from very long sequences.
    Still instructs the model to write a complete sentence.
    """
    if evidence and evidence.strip() and evidence.strip() != "nan":
        ev = evidence.strip()[:200]  # cap evidence length
        return f"Fix the incorrect fact. Evidence: {ev} Statement: {wrong} Correction:"
    else:
        return f"Fix the incorrect fact. Statement: {wrong} Correction:"

# ── Dataset ───────────────────────────────────────────────
class CorrectionDataset(Dataset):
    def __init__(self, df, tok, max_in, max_out):
        self.data = df.reset_index(drop=True)
        self.tok  = tok
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
            "input_ids"     : in_enc["input_ids"].squeeze(0),
            "attention_mask": in_enc["attention_mask"].squeeze(0),
            "labels"        : labels
        }

train_ds = CorrectionDataset(df_train, tokenizer, CONFIG["max_input_length"],
                              CONFIG["max_output_length"])
val_ds   = CorrectionDataset(df_val, tokenizer, CONFIG["max_input_length"],
                             CONFIG["max_output_length"])

train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                          num_workers=CONFIG["num_workers"], pin_memory=True,
                          persistent_workers=True)
val_loader   = DataLoader(val_ds, batch_size=CONFIG["batch_size"]*2, shuffle=False,
                          num_workers=CONFIG["num_workers"], pin_memory=True,
                          persistent_workers=True)

print(f"   Train batches: {len(train_loader)} | Val batches: {len(val_loader)}\n")

# ── Model — FP32 only ─────────────────────────────────────
print("--- Loading Flan-T5-small (FP32) ---")
model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"]).to(device)
model.gradient_checkpointing_enable()
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                  weight_decay=CONFIG["weight_decay"], eps=1e-8)
total_steps  = (len(train_loader) // CONFIG["grad_accum"]) * CONFIG["epochs"]
warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

print(f"   Epochs      : {CONFIG['epochs']} | LR: {CONFIG['learning_rate']}")
print(f"   Eff. batch  : {CONFIG['batch_size']*CONFIG['grad_accum']}")
print(f"   FP16        : {CONFIG['use_fp16']} (off to prevent NaN)\n")

# ── Train ─────────────────────────────────────────────────
def train_epoch(ep):
    model.train()
    total_loss = 0
    valid_steps = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        lbls  = batch["labels"].to(device)

        out   = model(input_ids=ids, attention_mask=mask, labels=lbls)
        loss  = out.loss

        # FIX 4: Skip NaN batches
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        (loss / CONFIG["grad_accum"]).backward()
        total_loss  += loss.item()
        valid_steps += 1

        if (step+1) % CONFIG["grad_accum"] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        if (step+1) % 100 == 0 and valid_steps > 0:
            print(f"   Ep {ep} | Step {step+1:>4}/{len(train_loader)} "
                  f"| Loss: {total_loss/valid_steps:.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e}")
            if torch.cuda.is_available():
                u = torch.cuda.memory_allocated()/1e9
                t = torch.cuda.get_device_properties(0).total_memory/1e9
                print(f"             GPU: {u:.2f}/{t:.1f} GB")

    return total_loss/valid_steps if valid_steps > 0 else float("nan")

def validate(num_gen=150):
    model.eval()
    total_loss, valid = 0, 0
    generated, references = [], []
    gen_count = 0

    with torch.no_grad():
        for batch in val_loader:
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["labels"].to(device)

            out = model(input_ids=ids, attention_mask=mask, labels=lbls)
            if not (torch.isnan(out.loss) or torch.isinf(out.loss)):
                total_loss += out.loss.item(); valid += 1

            if gen_count < num_gen:
                gen_ids = model.generate(
                    input_ids=ids, attention_mask=mask,
                    max_length           = CONFIG["max_output_length"],
                    min_length           = 10,
                    num_beams            = 4,
                    length_penalty       = 1.5,
                    early_stopping       = True,
                    no_repeat_ngram_size = 3
                )
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                lab   = lbls.clone()
                lab[lab==-100] = tokenizer.pad_token_id
                refs  = tokenizer.batch_decode(lab, skip_special_tokens=True)
                generated.extend(preds); references.extend(refs)
                gen_count += len(ids)

    avg_loss = total_loss/valid if valid>0 else float("nan")
    scores = {}
    if ROUGE_OK and generated:
        r = rouge_metric.compute(predictions=generated, references=references)
        scores = {"rouge1": round(r["rouge1"]*100,2),
                  "rouge2": round(r["rouge2"]*100,2),
                  "rougeL": round(r["rougeL"]*100,2)}
    return avg_loss, scores, generated[:3], references[:3]

def correct(wrong, evidence=""):
    model.eval()
    prompt = build_prompt(wrong, evidence)
    enc = tokenizer(prompt, max_length=CONFIG["max_input_length"],
                    truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids            = enc["input_ids"].to(device),
            attention_mask       = enc["attention_mask"].to(device),
            max_length           = CONFIG["max_output_length"],
            min_length           = 10,
            num_beams            = 4,
            length_penalty       = 1.5,
            early_stopping       = True,
            no_repeat_ngram_size = 3,
            do_sample            = False
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)

# ── Main loop ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60+"\n  TRAINING STARTED (FP32, NaN-safe)\n"+"="*60)

    best_rouge, patience_cnt = 0, 0
    tr_losses, vl_losses, rouge_hist = [], [], []

    for epoch in range(1, CONFIG["epochs"]+1):
        print(f"\n{'='*60}\n  Epoch {epoch}/{CONFIG['epochs']}\n{'='*60}")

        tr_loss = train_epoch(epoch)
        tr_losses.append(tr_loss)

        vl_loss, scores, sample_preds, sample_refs = validate()
        vl_losses.append(vl_loss)
        rouge_l = scores.get("rougeL", 0)
        rouge_hist.append(rouge_l)

        print(f"\n  Epoch {epoch}: TrainLoss={tr_loss:.4f} | ValLoss={vl_loss:.4f}")
        if scores:
            print(f"  ROUGE-1:{scores['rouge1']} | ROUGE-2:{scores['rouge2']} | ROUGE-L:{scores['rougeL']}")

        print(f"\n  Samples:")
        for p, r in zip(sample_preds, sample_refs):
            print(f"    Gen: {p[:100]}")
            print(f"    Ref: {r[:100]}\n")

        if rouge_l > best_rouge:
            best_rouge, patience_cnt = rouge_l, 0
            bp = os.path.join(CONFIG["output_dir"],"best")
            model.save_pretrained(bp); tokenizer.save_pretrained(bp)
            print(f"  ✅ Best saved — ROUGE-L:{best_rouge:.2f}")
        else:
            patience_cnt += 1
            print(f"  ⚠ No improvement ({patience_cnt}/{CONFIG['patience']})")
            if patience_cnt >= CONFIG["patience"]:
                print("  🛑 Early stopping"); break

        if torch.cuda.is_available(): torch.cuda.empty_cache()

    fp = os.path.join(CONFIG["output_dir"],"final")
    model.save_pretrained(fp); tokenizer.save_pretrained(fp)
    print(f"\n✅ Final model saved: {fp}")

    with open(os.path.join(CONFIG["output_dir"],"results.txt"),"w") as f:
        f.write("CORRECTION ENGINE — v3 (NaN Fixed)\n"+"="*50+"\n\n")
        f.write(f"Model: {CONFIG['model_name']}\n")
        f.write(f"FP16: {CONFIG['use_fp16']}\nLR: {CONFIG['learning_rate']}\n")
        f.write(f"Train: {len(df_train)}\nBest ROUGE-L: {best_rouge:.2f}\n")

    fig, axes = plt.subplots(1,2,figsize=(12,5))
    ep_d = range(1, len(tr_losses)+1)
    axes[0].plot(ep_d, tr_losses,"b-o",label="Train",lw=2)
    axes[0].plot(ep_d, vl_losses,"r-o",label="Val",lw=2)
    axes[0].set_title("Loss",fontweight="bold"); axes[0].legend()
    axes[0].grid(True,alpha=0.3)
    axes[1].plot(ep_d, rouge_hist,"g-o",label="ROUGE-L",lw=2)
    axes[1].axhline(y=30,color="orange",linestyle="--",label="Good (30)")
    axes[1].set_title("ROUGE-L",fontweight="bold"); axes[1].legend()
    axes[1].grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"],"training_loss.png"),dpi=150)
    print("✅ Plot saved")

    print("\n--- Testing ---")
    test_cases = [
        ("Albert Einstein invented the telephone.",
         "Alexander Graham Bell is credited with inventing the telephone in 1876."),
        ("The Amazon is the longest river in the world.",
         "The Nile is the longest river at 6,650 km. The Amazon is the largest by water flow."),
        ("Shakespeare was born in London.",
         "William Shakespeare was born in Stratford-upon-Avon, England in 1564."),
        ("Water boils at 50 degrees Celsius.",
         "Water boils at 100 degrees Celsius at standard atmospheric pressure."),
        ("1 liter is equal to 1200 ml.",
         "1 liter is equal to 1000 milliliters."),
        ("Domino's is a sushi restaurant.",
         "Domino's Pizza is an American pizza restaurant chain."),
    ]
    for i, (wrong, ev) in enumerate(test_cases, 1):
        corrected = correct(wrong, ev)
        print(f"  {i}. ❌ {wrong}")
        print(f"     ✅ {corrected}\n")

    print("="*60+f"\n  PHASE 6 COMPLETE | Best ROUGE-L: {best_rouge:.2f}\n"+"="*60)
