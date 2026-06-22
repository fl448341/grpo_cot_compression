# parsing the run log (vibecoded with claude)

import csv
import re
import sys


PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_/]*):(-?\d+\.?\d*)")
LAM = re.compile(r"TRAIN lambda=(\S+)")

COLS = ["lambda", "step", "response_length_mean", "response_length_clip_ratio",
        "critic_score_mean", "actor_kl_loss", "val_test_score"]
KEYMAP = {
    "response_length_mean": "response_length/mean",
    "response_length_clip_ratio": "response_length/clip_ratio",
    "critic_score_mean": "critic/score/mean",
    "actor_kl_loss": "actor/kl_loss",
    "val_test_score": "val/test_score/openai/gsm8k",
}


def main():
    log_path = sys.argv[1] if len(sys.argv) > 1 else "runpod_results/run.log"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "phase2/results/train_trajectory.csv"

    cur_lam = None
    rows = []
    for line in open(log_path, errors="replace"):
        m = LAM.search(line)
        if m:
            cur_lam = m.group(1)
            continue
        if "step:" not in line or "response_length/mean" not in line:
            continue
        d = dict(PAIR.findall(line))
        row = {"lambda": cur_lam, "step": d.get("step", "")}
        for col, key in KEYMAP.items():
            row[col] = d.get(key, "")
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    print(f"Zapisano {len(rows)} krokow -> {out_path}\n")
    print(f"{'lambda':>8} {'last_step':>9} {'resp_len':>9} {'clip':>6} {'score':>7} {'val_acc':>8}")
    last = {}
    for r in rows:
        last[r["lambda"]] = r
    for lam, r in last.items():
        print(f"{str(lam):>8} {r['step']:>9} {r['response_length_mean']:>9} "
              f"{r['response_length_clip_ratio']:>6} {r['critic_score_mean']:>7} "
              f"{r['val_test_score'] or '-':>8}")


if __name__ == "__main__":
    main()
