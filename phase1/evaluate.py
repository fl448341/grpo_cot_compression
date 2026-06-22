from .extract import extract_last_number
from .model import generate_batch


def run_eval(model, tokenizer, system_prompt, eval_dataset, max_new_tokens=512,
             batch_size=16, verbose=False):
    questions = [item["question"] for item in eval_dataset]
    truths = [item["answer"].split("####")[-1].strip().replace(",", "") for item in eval_dataset]

    outputs = generate_batch(
        model, tokenizer,
        system_instruction=system_prompt,
        user_queries=questions,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        do_sample=False,
    )

    records = []
    correct = 0
    for question, ground_truth, out in zip(questions, truths, outputs):
        pred = extract_last_number(out)
        is_correct = (pred == ground_truth)
        correct += is_correct

        records.append({
            "is_correct": bool(is_correct),
            "tokens": len(tokenizer.encode(out)),
            "pred": pred,
            "truth": ground_truth,
            "question": question,
            "output": out,
        })
        if verbose:
            print(f"  truth={ground_truth} pred={pred} ok={is_correct}", flush=True)

    n = len(records)
    acc = correct / n
    avg_len = sum(r["tokens"] for r in records) / n
    correct_lens = [r["tokens"] for r in records if r["is_correct"]]
    avg_len_correct = sum(correct_lens) / len(correct_lens) if correct_lens else float("nan")

    return {
        "accuracy": acc,
        "avg_tokens": avg_len,
        "avg_tokens_correct": avg_len_correct,
        "n": n,
        "records": records,
    }
