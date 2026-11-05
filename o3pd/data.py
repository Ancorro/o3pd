"""Benchmarks (HMMT25, AIME24, AIME25) and OpenThoughts math training prompts."""
import json
import os
import random

PROMPT = "Problem: {problem}\nReason step by step and put the final answer in \\boxed{{}}."

BENCHMARKS = {
    "hmmt25": ("MathArena/hmmt_feb_2025", "train"),
    "aime24": ("HuggingFaceH4/aime_2024", "train"),
    "aime25": ("MathArena/aime_2025", "train"),
}


def load_benchmark(name, cache_dir="data"):
    """List of {"problem", "answer"}; cached as JSONL so later runs need no network."""
    path = os.path.join(cache_dir, f"{name}.jsonl")
    if os.path.exists(path):
        return [json.loads(line) for line in open(path, encoding="utf-8")]
    from datasets import load_dataset
    repo, split = BENCHMARKS[name]
    rows = [{"problem": r["problem"], "answer": str(r["answer"])} for r in load_dataset(repo, split=split)]
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def load_train_prompts(n=3200, seed=0, cache_dir="data", max_rows=None):
    """n math problems from OpenThoughts3 (domain == "math"), deduplicated, shuffled with `seed`.

    The paper says "the mathematical reasoning subset of OpenThoughts" and gives no size; 100 steps x batch 32
    uses 3,200 prompts. We stream the dataset and keep unique math problems until we have enough.
    """
    path = os.path.join(cache_dir, f"openthoughts_math_{n}_s{seed}.jsonl")
    if os.path.exists(path):
        return [json.loads(line)["problem"] for line in open(path, encoding="utf-8")]
    from datasets import load_dataset
    ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", streaming=True)
    seen, pool = set(), []
    want = max_rows or n * 4
    for r in ds:
        if r.get("domain") != "math":
            continue
        q = r["conversations"][0]["value"].strip()
        if q in seen:
            continue
        seen.add(q)
        pool.append(q)
        if len(pool) >= want:
            break
    random.Random(seed).shuffle(pool)
    pool = pool[:n]
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in pool:
            f.write(json.dumps({"problem": q}) + "\n")
    return pool


def last_boxed(text):
    """Contents of the last \\boxed{...} in text (brace-balanced), or None."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def is_correct(response, answer):
    """Compare the last \\boxed{} answer with the gold answer (math-verify, with a string fallback)."""
    pred = last_boxed(response)
    if pred is None:
        return False
    if pred.strip().replace(" ", "") == answer.strip().replace(" ", ""):
        return True
    try:
        from math_verify import parse, verify
        return bool(verify(parse("$" + answer + "$"), parse("$" + pred + "$")))
    except Exception:
        return False
