"""Avg@k on HMMT25, AIME24, AIME25 with vLLM (paper Table 5 settings).

python -m o3pd.evaluate --model Qwen/Qwen3-1.7B [--lora runs/x/step_20] --out results/eval/base_1.7b.json
"""
import argparse
import json
import os
import time

from .data import PROMPT, is_correct, load_benchmark


def evaluate(model, lora=None, benchmarks=("hmmt25", "aime24", "aime25"), seeds=(41, 42, 43), n_per_seed=4,
             max_tokens=38912, max_model_len=40960, thinking=True, temperature=1.0, top_p=1.0,
             gpu_mem=0.9, limit=None, keep_text=False):
    from vllm import LLM, SamplingParams
    kw = dict(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_mem, seed=0)
    if lora:
        kw.update(enable_lora=True, max_lora_rank=64)
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    lora_req = None
    if lora:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("student", 1, lora)
    out = {"model": model, "lora": lora, "thinking": thinking, "max_tokens": max_tokens, "seeds": list(seeds),
           "n_per_seed": n_per_seed, "temperature": temperature, "top_p": top_p, "benchmarks": {}}
    t0 = time.time()
    for b in benchmarks:
        probs = load_benchmark(b)[:limit]
        prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=p["problem"])}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
                   for p in probs]
        correct = [[] for _ in probs]
        lengths, truncated = [], 0
        for s in seeds:
            sp = SamplingParams(n=n_per_seed, temperature=temperature, top_p=top_p, top_k=-1, min_p=0.0,
                                max_tokens=max_tokens, seed=s)
            res = llm.generate(prompts, sp, lora_request=lora_req)
            for i, r in enumerate(res):
                for o in r.outputs:
                    correct[i].append(is_correct(o.text, probs[i]["answer"]))
                    lengths.append(len(o.token_ids))
                    truncated += o.finish_reason == "length"
        acc = 100 * sum(sum(c) for c in correct) / sum(len(c) for c in correct)
        out["benchmarks"][b] = {"acc": acc, "per_problem": [sum(c) / len(c) for c in correct],
                                "mean_len": sum(lengths) / len(lengths), "truncated": truncated,
                                "n": sum(len(c) for c in correct)}
        print(f"{b}: {acc:.1f}  (mean len {out['benchmarks'][b]['mean_len']:.0f}, truncated {truncated})", flush=True)
    out["average"] = sum(v["acc"] for v in out["benchmarks"].values()) / len(out["benchmarks"])
    out["seconds"] = time.time() - t0
    print(f"average: {out['average']:.1f}  ({out['seconds'] / 60:.1f} min)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=38912)
    ap.add_argument("--n-per-seed", type=int, default=4)
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    a = ap.parse_args()
    r = evaluate(a.model, a.lora, thinking=not a.no_thinking, max_tokens=a.max_tokens, n_per_seed=a.n_per_seed,
                 seeds=tuple(int(s) for s in a.seeds.split(",")), limit=a.limit, gpu_mem=a.gpu_mem,
                 max_model_len=max(a.max_tokens + 2048, 4096))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(r, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
