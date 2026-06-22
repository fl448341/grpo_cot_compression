MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"

N_EVAL = 200

MAX_NEW_TOKENS = 512

BATCH_SIZE = 16

PROMPTS = {
    "baseline (natural)": "",
    "brief":   "Solve the math problem. Think step by step, but keep your reasoning brief.",
    "concise": "Solve the math problem using as few reasoning steps as possible. Be concise, then give the final answer.",
    "minimal": "Solve the math problem. Give only the minimal reasoning needed, then state the final answer.",
}
