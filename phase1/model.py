import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if torch.cuda.is_available():
        print(f"Model loaded. GPU usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print("Model loaded on CPU.")
    return model, tokenizer


def generate_batch(model, tokenizer, system_instruction: str, user_queries: list,
                   max_new_tokens: int = 512, batch_size: int = 16, **generation_kwargs):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs = []
    for start in range(0, len(user_queries), batch_size):
        chunk = user_queries[start:start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_instruction},
                 {"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True,
            )
            for q in chunk
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **generation_kwargs,
        )
        gen_only = generated[:, inputs.input_ids.shape[1]:]
        outputs.extend(tokenizer.batch_decode(gen_only, skip_special_tokens=True))
        print(f"  ...generated {min(start + batch_size, len(user_queries))}/{len(user_queries)}", flush=True)

    return outputs


def generate_response_raw(model, tokenizer, system_instruction: str, user_query: str,
                          max_new_tokens: int = 1024, **generation_kwargs):
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_query},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **generation_kwargs,
    )

    raw_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

    isolated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    isolated_response = tokenizer.batch_decode(isolated_ids, skip_special_tokens=True)[0]

    return raw_response, isolated_response
