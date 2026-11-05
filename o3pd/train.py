"""On-policy distillation with LoRA: O3PD (order-only losses) or OPD (top-K JSD).

One step: sample a batch of prompts, the student generates one response each with vLLM (non-thinking mode,
up to 1,024 tokens), the frozen teacher scores every prefix of each response (thinking-mode chat template),
we keep its top-K tokens in order, and the student takes one LoRA step on the chosen loss.

python -m o3pd.train --student Qwen/Qwen3-1.7B --teacher Qwen/Qwen3-4B --loss adjacent --out runs/4b-1.7b-o3pd
"""
import argparse
import dataclasses
import json
import os
import random
import time

import torch

from .data import PROMPT, load_train_prompts
from .losses import LOSSES, agreement_metrics


@dataclasses.dataclass
class TrainConfig:
    student: str = "Qwen/Qwen3-1.7B"
    teacher: str = "Qwen/Qwen3-4B"
    loss: str = "adjacent"           # adjacent (O3PD) | logistic | topweighted | listmle | opd
    k: int = 8
    teacher_temperature: float = 1.0  # only used by opd; order-only losses ignore it
    steps: int = 100
    batch_size: int = 32
    lr: float = 5e-6
    lora_r: int = 64
    lora_alpha: int = 128
    max_grad_norm: float = 1.0
    rollout_tokens: int = 1024
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    seed: int = 42
    n_prompts: int = 3200
    save_every: int = 20
    metrics_every: int = 5
    student_device: str = "cuda:0"
    teacher_device: str = "cuda:0"
    vllm_mem: float = 0.35
    out: str = "runs/debug"
    tiny: bool = False  # CPU smoke test: tiny random models, HF generate instead of vLLM


def chat(tok, text, thinking):
    return tok.apply_chat_template([{"role": "user", "content": PROMPT.format(problem=text)}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)


def response_logits(model, prompt_ids, resp_ids, device):
    """Logits predicting each response token: shape (len(resp), V)."""
    ids = torch.tensor([prompt_ids + resp_ids], device=device)
    logits = model(input_ids=ids).logits[0]
    p = len(prompt_ids)
    return logits[p - 1:p - 1 + len(resp_ids)]


class Rollouts:
    """Student sampling. vLLM with the current LoRA adapter, or HF generate for the tiny CPU test."""

    def __init__(self, cfg, model, tok):
        self.cfg, self.model, self.tok = cfg, model, tok
        self.llm = None
        if not cfg.tiny:
            from vllm import LLM
            self.llm = LLM(model=cfg.student, enable_lora=True, max_lora_rank=cfg.lora_r, max_loras=1,
                           gpu_memory_utilization=cfg.vllm_mem, max_model_len=cfg.rollout_tokens + 1024,
                           seed=cfg.seed, enable_prefix_caching=False)

    def generate(self, prompts, step, adapter_dir):
        if self.llm is None:
            out = []
            for p in prompts:
                ids = self.tok(p, return_tensors="pt").input_ids
                g = self.model.generate(ids, max_new_tokens=self.cfg.rollout_tokens, do_sample=True,
                                        temperature=self.cfg.rollout_temperature, pad_token_id=self.tok.eos_token_id)
                out.append(g[0, ids.shape[1]:].tolist())
            return out
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        sp = SamplingParams(n=1, temperature=self.cfg.rollout_temperature, top_p=self.cfg.rollout_top_p, top_k=-1,
                            max_tokens=self.cfg.rollout_tokens, seed=self.cfg.seed + step)
        res = self.llm.generate(prompts, sp, lora_request=LoRARequest(f"s{step}", step + 1, adapter_dir),
                                use_tqdm=False)
        return [list(r.outputs[0].token_ids) for r in res]


def load_models(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(cfg.student)
    dt = torch.float32 if cfg.tiny else torch.bfloat16
    student = AutoModelForCausalLM.from_pretrained(cfg.student, dtype=dt).to(cfg.student_device)
    student.gradient_checkpointing_enable()
    student.enable_input_require_grads()
    student = get_peft_model(student, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    teacher = AutoModelForCausalLM.from_pretrained(cfg.teacher, dtype=dt).to(cfg.teacher_device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return tok, student, teacher


def train(cfg, prompts=None):
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.out, exist_ok=True)
    json.dump(dataclasses.asdict(cfg), open(os.path.join(cfg.out, "config.json"), "w"), indent=1)
    prompts = prompts or load_train_prompts(cfg.n_prompts, seed=0)
    tok, student, teacher = None, None, None
    if not cfg.tiny:
        rollouts = Rollouts(cfg, None, None)  # vLLM first, so it sees free memory before HF models load
    tok, student, teacher = load_models(cfg)
    if cfg.tiny:
        rollouts = Rollouts(cfg, student, tok)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.0)
    loss_fn = LOSSES[cfg.loss]
    adapter_dir = os.path.join(cfg.out, "adapter_live")
    log = open(os.path.join(cfg.out, "log.jsonl"), "a")
    order = list(range(len(prompts)))
    random.shuffle(order)
    for step in range(cfg.steps):
        t0 = time.time()
        batch = [prompts[order[(step * cfg.batch_size + i) % len(order)]] for i in range(cfg.batch_size)]
        s_prompts = [chat(tok, q, thinking=False) for q in batch]
        t_prompts = [chat(tok, q, thinking=True) for q in batch]
        student.save_pretrained(adapter_dir)
        student.eval()
        resps = rollouts.generate(s_prompts, step, adapter_dir)
        student.train()
        t_gen = time.time() - t0
        opt.zero_grad(set_to_none=True)
        tot_loss, n_used, metrics = 0.0, 0, []
        for sp, tp, r in zip(s_prompts, t_prompts, resps):
            if len(r) == 0:
                continue
            n_used += 1
            sp_ids = tok(sp, add_special_tokens=False).input_ids
            tp_ids = tok(tp, add_special_tokens=False).input_ids
            with torch.no_grad():
                t_logits = response_logits(teacher, tp_ids, r, cfg.teacher_device).float()
                t_lp, t_idx = torch.log_softmax(t_logits, -1).topk(cfg.k, -1)  # sorted descending
                del t_logits
            t_lp, t_idx = t_lp.to(cfg.student_device), t_idx.to(cfg.student_device)
            s_logits = response_logits(student, sp_ids, r, cfg.student_device)
            s_at = s_logits.gather(-1, t_idx).float()
            per_pos = loss_fn(s_at, t_lp, cfg.teacher_temperature)
            seq_loss = per_pos.mean()
            (seq_loss / cfg.batch_size).backward()
            tot_loss += seq_loss.item()
            if step % cfg.metrics_every == 0:
                metrics.append(agreement_metrics(s_logits.detach(), t_idx, t_lp, k=min(8, cfg.k)))
            del s_logits, s_at
        gn = torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], cfg.max_grad_norm)
        opt.step()
        rec = {"step": step + 1, "loss": tot_loss / max(n_used, 1), "grad_norm": gn.item(),
               "mean_resp_len": sum(map(len, resps)) / len(resps),
               "frac_truncated": sum(len(r) >= cfg.rollout_tokens for r in resps) / len(resps),
               "sec_gen": t_gen, "sec_step": time.time() - t0}
        if metrics:
            for k in metrics[0]:
                rec[k] = sum(m[k] for m in metrics) / len(metrics)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        if (step + 1) % cfg.save_every == 0 or step + 1 == cfg.steps:
            student.save_pretrained(os.path.join(cfg.out, f"step_{step + 1}"))
    return student


def main():
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(TrainConfig):
        if f.type == "bool" or isinstance(f.default, bool):
            ap.add_argument("--" + f.name.replace("_", "-"), action="store_true")
        else:
            ap.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    train(TrainConfig(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
