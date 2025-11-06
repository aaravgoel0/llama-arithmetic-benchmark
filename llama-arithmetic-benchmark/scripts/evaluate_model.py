import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import csv
import re
import math
import random
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 8
BATCH_SIZE = 8
PREVIEW_SAMPLES = 0


def load_problems(filename):
    path = filename if filename.startswith("data/") else os.path.join("data", filename)
    problems = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            problems.append({"problem": row["problem"], "answer": float(row["answer"])})
    return problems

def write_results_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["dataset", "problem", "expected", "predicted", "is_correct", "model_raw_output"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}")

def write_summary_csv(path, summary_rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["dataset", "num_samples", "num_correct", "accuracy"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved: {path}")

def extract_leading_number(text):
    m = re.match(r"\s*([+-]?\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else math.nan

def make_chat_inputs(tokenizer, expr, device_str="cuda"):
    messages = [
        {"role": "system",
         "content": "You are a calculator. Output ONLY the final numeric value. No words, no labels, no steps, no punctuation, no units."},
        {"role": "user",
         "content": f"{expr}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").to(device_str)

def solve_batch(model, tokenizer, problems, device_str="cuda"):
    encoded_list = [make_chat_inputs(tokenizer, p["problem"] + " = ?", device_str) for p in problems]
    input_ids_list = [enc["input_ids"][0] for enc in encoded_list]
    attn_list = [enc["attention_mask"][0] for enc in encoded_list]
    max_len = max(x.size(0) for x in input_ids_list)

    def pad_to(t, L, pad_id):
        if t.size(0) == L:
            return t
        pad = torch.full((L - t.size(0),), pad_id, dtype=t.dtype, device=t.device)
        return torch.cat([t, pad], dim=0)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batched_input_ids = torch.stack([pad_to(t, max_len, pad_id) for t in input_ids_list], dim=0)
    batched_attention = torch.stack([pad_to(t, max_len, 0) for t in attn_list], dim=0)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=batched_input_ids,
            attention_mask=batched_attention,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0
        )

    new_tokens = outputs[:, batched_input_ids.shape[1]:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    preds = []
    for txt in decoded:
        first_line = (txt or "").splitlines()[0]
        preds.append((extract_leading_number(first_line), first_line))
    return preds


def evaluate_and_log(dataset_name, model, tokenizer, problems, device_str="cuda",
                     preview_samples=PREVIEW_SAMPLES, csv_path=None):
    rows = []
    correct = 0
    total = len(problems)
    print(f"\n===== {dataset_name.upper()} — Evaluating all {total} problems =====")

    for i in range(0, total, BATCH_SIZE):
        batch = problems[i:i+BATCH_SIZE]
        batch_preds = solve_batch(model, tokenizer, batch, device_str)
        for p, (pred, raw_text) in zip(batch, batch_preds):
            ok = (not math.isnan(pred)) and (abs(pred - p["answer"]) < 1e-6)
            correct += int(ok)
            rows.append({
                "dataset": dataset_name,
                "problem": p["problem"],
                "expected": p["answer"],
                "predicted": (None if math.isnan(pred) else pred),
                "is_correct": bool(ok),
                "model_raw_output": raw_text
            })

        done = min(i + BATCH_SIZE, total)
        print(f"  Progress: {done}/{total} ({done/total:.0%})", end="\r")

    print()
    acc = correct / total if total else 0.0
    if csv_path:
        write_results_csv(csv_path, rows)

    k = min(preview_samples, total)
    if k > 0:
        sample_rows = random.sample(rows, k)
        print(f"\n-- Preview ({k} of {total}) --")
        for i, r in enumerate(sample_rows, 1):
            status = "✓ CORRECT" if r["is_correct"] else "✗ INCORRECT"
            print(f"[{i}/{k}] {r['problem']} -> Pred: {r['predicted']} | Exp: {r['expected']} | {status}")

    print(f"{dataset_name} accuracy: {correct}/{total} ({acc:.2%})")
    return acc, correct, total, rows

def save_accuracy_chart(summary_rows, out_path="accuracy_chart.png"):
    labels = [r["dataset"] for r in summary_rows]
    accuracies = [float(r["accuracy"]) for r in summary_rows]

    plt.figure(figsize=(5, 4))
    plt.bar(labels, [a * 100 for a in accuracies])
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy by Dataset")
    for i, a in enumerate(accuracies):
        plt.text(i, a * 100 + 1, f"{a*100:.1f}%", ha="center", va="bottom")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart: {out_path}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (FP16 weights with CPU/disk offload; no quantization).")

    print(f"Loading {MODEL_NAME} (FP16, no quantization; fast safe T4 offload)...")

    hf_token = os.getenv("HF_TOKEN", None)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    gpu_budget_gib = 11.0
    cpu_budget_gib = 8.0
    max_memory = {0: f"{gpu_budget_gib}GiB", "cpu": f"{cpu_budget_gib}GiB"}
    os.makedirs("offload", exist_ok=True)

    print("Initializing model on CPU first, then streaming to GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
        offload_folder="offload",
        use_safetensors=True,
        resume_download=True,
        local_files_only=False
    )
    model.eval()
    print(" Model loaded (FP16, no quantization). Using GPU+CPU offload to fit T4.")

    if not (os.path.exists("data/integer_arithmetic.csv") and os.path.exists("data/decimal_arithmetic.csv")):
        print("Datasets not found. Running generate_datasets.py ...")
        import generate_datasets
        generate_datasets.main()

    int_probs = load_problems("integer_arithmetic.csv")
    dec_probs = load_problems("decimal_arithmetic.csv")
    device_str = "cuda"

    int_acc, int_correct, int_total, _ = evaluate_and_log(
        "integer",
        model, tok, int_probs,
        device_str=device_str,
        preview_samples=PREVIEW_SAMPLES,
        csv_path="results_integer.csv"
    )

    dec_acc, dec_correct, dec_total, _ = evaluate_and_log(
        "decimal",
        model, tok, dec_probs,
        device_str=device_str,
        preview_samples=PREVIEW_SAMPLES,
        csv_path="results_decimal.csv"
    )

    summary_rows = [
        {"dataset": "integer", "num_samples": int_total, "num_correct": int_correct, "accuracy": f"{int_acc:.4f}"},
        {"dataset": "decimal", "num_samples": dec_total, "num_correct": dec_correct, "accuracy": f"{dec_acc:.4f}"},
    ]
    write_summary_csv("results_summary.csv", summary_rows)

    save_accuracy_chart(
        [{"dataset": "integer", "accuracy": float(int_acc)},
         {"dataset": "decimal", "accuracy": float(dec_acc)}],
        out_path="accuracy_chart.png"
    )

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Integer Arithmetic Accuracy: {int_acc:.2%}  ({int_correct}/{int_total})")
    print(f"Decimal  Arithmetic Accuracy: {dec_acc:.2%}  ({dec_correct}/{dec_total})")
    print("CSV files: results_integer.csv, results_decimal.csv, results_summary.csv")
    print("Chart: accuracy_chart.png")

if __name__ == "__main__":
    main()
