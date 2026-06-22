import argparse
import csv
import json
import os

from datasets import load_dataset

from . import config
from .evaluate import run_eval
from .model import load_model
from .plot import plot_length_bins, plot_tradeoff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=config.MODEL_ID)
    parser.add_argument("--n-eval", type=int, default=config.N_EVAL)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--out", default="phase1/results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Model: {args.model_id} | N_EVAL: {args.n_eval} | max_new_tokens: {args.max_new_tokens}")
    model, tokenizer = load_model(args.model_id)

    eval_set = load_dataset(config.DATASET_ID, config.DATASET_CONFIG, split=f"test[0:{args.n_eval}]")

    results = {}
    for name, system_prompt in config.PROMPTS.items():
        print(f"\n=== Wariant: {name!r} ===", flush=True)
        res = run_eval(model, tokenizer, system_prompt, eval_set,
                       max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
                       verbose=args.verbose)
        results[name] = res
        print(f"  accuracy={res['accuracy']:.2%} | avg_tokens={res['avg_tokens']:.1f} | "
              f"avg_tokens(correct)={res['avg_tokens_correct']:.1f}", flush=True)

    print(f"\n{'wariant':<22}{'accuracy':>10}{'avg_tok':>10}{'avg_tok(ok)':>14}")
    for name, r in results.items():
        print(f"{name:<22}{r['accuracy']:>9.1%}{r['avg_tokens']:>10.1f}{r['avg_tokens_correct']:>14.1f}")

    csv_path = os.path.join(args.out, "phase1_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "system_prompt", "n", "accuracy", "avg_tokens", "avg_tokens_correct"])
        for name, r in results.items():
            w.writerow([name, config.PROMPTS[name], r["n"],
                        f"{r['accuracy']:.4f}", f"{r['avg_tokens']:.2f}", f"{r['avg_tokens_correct']:.2f}"])
    print(f"\nSaved metrics: {csv_path}")

    records_path = os.path.join(args.out, "phase1_records.json")
    with open(records_path, "w") as f:
        json.dump({name: r["records"] for name, r in results.items()}, f, indent=1)
    print(f"Saved records: {records_path}")

    plot_tradeoff(results, os.path.join(args.out, "phase1_tradeoff.png"))
    baseline_key = next(iter(config.PROMPTS))  # "baseline (natural)"
    plot_length_bins(results[baseline_key]["records"], os.path.join(args.out, "phase1_length_bins.png"))
    print(f"Saved plots to: {args.out}/")


if __name__ == "__main__":
    main()
