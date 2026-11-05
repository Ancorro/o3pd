"""Two training steps end to end on tiny random Qwen3 models (CPU, no download beyond the tokenizer)."""
import json
import os

import pytest
import torch

from o3pd.train import TrainConfig, train


def make_tiny(tmp, seed, layers):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    tok_dir = os.path.join(os.path.dirname(__file__), "..", "data", "tokenizer")
    try:
        tok = AutoTokenizer.from_pretrained(tok_dir if os.path.isdir(tok_dir) else "Qwen/Qwen3-0.6B")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"tokenizer unavailable: {e}")
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=512)
    m = Qwen3ForCausalLM(cfg)
    path = os.path.join(tmp, f"m{seed}")
    m.save_pretrained(path)
    tok.save_pretrained(path)
    return path


@pytest.mark.parametrize("loss", ["adjacent", "opd"])
def test_two_steps(tmp_path, loss):
    torch.set_num_threads(2)
    s = make_tiny(str(tmp_path), 0, 1)
    t = make_tiny(str(tmp_path), 1, 2)
    out = str(tmp_path / f"run_{loss}")
    cfg = TrainConfig(student=s, teacher=t, loss=loss, steps=2, batch_size=2, rollout_tokens=8, lr=1e-2,
                      lora_r=4, lora_alpha=8, student_device="cpu", teacher_device="cpu", save_every=2,
                      metrics_every=1, out=out, tiny=True)
    train(cfg, prompts=["What is 1+1?", "What is 2+3?"])
    log = [json.loads(line) for line in open(os.path.join(out, "log.jsonl"))]
    assert len(log) == 2 and all(r["loss"] > 0 for r in log)
    assert "kendall_tau" in log[0]
    assert os.path.exists(os.path.join(out, "step_2", "adapter_config.json"))
