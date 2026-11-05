# o3pd

*AI-sourced: assisted by AI agents.*

An independent reproduction of **"Order-Only On-Policy Distillation"** (O3PD) by Guangzeng Han, Yu Wu and
Xiaolei Huang ([arXiv:2611.02112](https://arxiv.org/abs/2611.02112), 2026). The authors released no code; everything
here was written from the paper. It is not affiliated with the authors.

On-policy distillation (OPD) trains a student LLM on its own sampled responses by matching the teacher's token
probabilities at every prefix. O3PD keeps the same on-policy setup but trains the student only to reproduce the
*order* of the teacher's top-K tokens, through an adjacent pairwise logistic loss. The paper reports that this beats
OPD on HMMT25, AIME24 and AIME25 for six Qwen3 teacher–student pairs, needs no teacher-temperature tuning, and keeps
improving where OPD degrades.

Who it is for: anyone checking the paper's claims or wanting a small, readable on-policy distillation loop for
Hugging Face models.

Built with: Python 3.12, PyTorch, Transformers, PEFT (LoRA), vLLM.

## How to run

```
bash setup.sh                       # PyPI only; CPU is enough for the tests
python -m pytest -q                 # loss tests + a 2-step training run on tiny random Qwen3 models, CPU
# GPU (Linux, vLLM):
python -m o3pd.evaluate --model Qwen/Qwen3-1.7B --out results/eval/base_1.7b.json
python -m o3pd.train --student Qwen/Qwen3-1.7B --teacher Qwen/Qwen3-4B --loss adjacent --out runs/4b-1.7b-o3pd
python -m o3pd.train --student Qwen/Qwen3-1.7B --teacher Qwen/Qwen3-4B --loss opd --out runs/4b-1.7b-opd
python -m o3pd.evaluate --model Qwen/Qwen3-1.7B --lora runs/4b-1.7b-o3pd/step_100 --out results/eval/o3pd_100.json
```

## Code

- `o3pd/losses.py`: the order-only losses of paper Table 2 (adjacent logistic = O3PD, all-pairs logistic,
  top-weighted logistic, ListMLE), the OPD baseline (JSD on the teacher's top-K, renormalized), and the Figure 5
  agreement metrics (JSD@top-8, Kendall's tau, top-8 overlap, top-1 agreement, student entropy).
- `o3pd/train.py`: the on-policy loop. vLLM samples one non-thinking response per prompt from the current LoRA
  student; the frozen teacher scores every prefix under its thinking-mode chat template; the student takes a LoRA step.
- `o3pd/evaluate.py`: Avg@12 on HMMT25, AIME24, AIME25 (4 samples under each of seeds 41, 42, 43; thinking mode).
- `o3pd/data.py`: benchmark loading, OpenThoughts math prompts, `\boxed{}` answer checking.

## Results

In progress.

## Sources and attribution

- Paper: Han, Wu, Huang, "Order-Only On-Policy Distillation", arXiv:2611.02112 (2026).
- Evaluation protocol and baselines: Zhao et al., "Self-Distilled Reasoner: On-Policy Self-Distillation for Large
  Language Models", arXiv:2601.18734; code https://github.com/siyan-zhao/OPSD (read for the thinking-mode setup; no
  code copied).
- Models: Qwen3-1.7B, Qwen3-4B, Qwen3-8B (Qwen team, Apache-2.0), https://huggingface.co/Qwen. `data/tokenizer` is
  the Qwen3 tokenizer (Apache-2.0), kept for the CPU tests.
- Training prompts: OpenThoughts3-1.2M math subset (Guha et al., Apache-2.0),
  https://huggingface.co/datasets/open-thoughts/OpenThoughts3-1.2M, downloaded by `o3pd/data.py`.
- Benchmarks (downloaded, not committed): MathArena/hmmt_feb_2025 and MathArena/aime_2025 (CC BY-NC-SA 4.0),
  HuggingFaceH4/aime_2024.
- Answer checking: math-verify (Hugging Face, Apache-2.0).

## Plans

See [ROADMAP.md](ROADMAP.md).

## Repo rules

This repo can be made public at any time, history included.

- No secrets: keys go in `.env` (untracked); `.env.example` lists the names only.
- No private information: cluster paths, host names, run logs, costs and personal notes go in `private/`, which is untracked and backed up outside git.
- Large data stays out of git; a script downloads it.
- Sources: add each new source to "Sources and attribution" in the commit that starts using it.
- Keep the real history: no squashing or force-pushing published commits.
