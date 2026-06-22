import argparse
import csv
import json
import os

from datasets import load_dataset

from phase1.extract import extract_last_number
from phase1.model import generate_batch, load_model

# same as in tinyzero training
GSM8K_INSTRUCTION = "Let's think step by step and output the final answer after \"####\"."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint path")
    ap.add_argument("--label", required=True, help="run label")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="phase2/results")
    ap.add_argument("--lam", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--lr", default="")
    ap.add_argument("--steps", default="")
    ap.add_argument("--grpo-n", default="")
    ap.add_argument("--temperature", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, tokenizer = load_model(args.ckpt)
    ds = load_dataset("openai/gsm8k", "main", split=f"test[0:{args.n_eval}]")

    questions = [item["question"] + " " + GSM8K_INSTRUCTION for item in ds]
    truths = [item["answer"].split("####")[-1].strip().replace(",", "") for item in ds]

    outputs = generate_batch(
        model, tokenizer, system_instruction="", user_queries=questions,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, do_sample=False,
    )

    records = []
    correct = n_hash = n_boxed = 0
    for q, truth, out in zip(questions, truths, outputs):
        pred = extract_last_number(out)
        ok = (pred == truth)
        correct += ok
        has_hash = "####" in out
        has_boxed = "\\boxed" in out
        n_hash += has_hash
        n_boxed += has_boxed
        records.append({
            "is_correct": bool(ok), "tokens": len(tokenizer.encode(out)),
            "pred": pred, "truth": truth, "has_hash": has_hash, "has_boxed": has_boxed,
            "question": q, "output": out,
        })

    import statistics
    n = len(records)
    acc = correct / n
    toks = sorted(r["tokens"] for r in records)
    avg_tok = sum(toks) / n
    med_tok = statistics.median(toks)
    pct_le128 = sum(t <= 128 for t in toks) / n # distribution by cot len
    pct_le256 = sum(t <= 256 for t in toks) / n
    pct_at_max = sum(t >= args.max_new_tokens for t in toks) / n
    pct_hash, pct_boxed = n_hash / n, n_boxed / n
    print(f"\n[{args.label}] acc={acc:.2%} | avg_tok={avg_tok:.0f} med={med_tok:.0f} | "
          f"<=128:{pct_le128:.0%} <=256:{pct_le256:.0%} =max:{pct_at_max:.0%} | %####={pct_hash:.0%}")

    csv_path = os.path.join(args.out, "phase2_eval.csv")
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["label", "lambda", "variant", "lr", "steps", "grpo_n", "temperature",
                        "max_resp", "n_eval", "accuracy", "avg_tokens", "median_tokens",
                        "pct_len_le128", "pct_len_le256", "pct_at_max", "pct_hash", "pct_boxed", "ckpt"])
        w.writerow([args.label, args.lam, args.variant, args.lr, args.steps, args.grpo_n,
                    args.temperature, args.max_new_tokens, n, f"{acc:.4f}", f"{avg_tok:.2f}",
                    f"{med_tok:.1f}", f"{pct_le128:.4f}", f"{pct_le256:.4f}", f"{pct_at_max:.4f}",
                    f"{pct_hash:.4f}", f"{pct_boxed:.4f}", args.ckpt])

    json.dump(records, open(os.path.join(args.out, f"records_{args.label}.json"), "w"), indent=1)
    print(f"Appended to {csv_path} | records: {args.out}/records_{args.label}.json")


if __name__ == "__main__":
    main()
