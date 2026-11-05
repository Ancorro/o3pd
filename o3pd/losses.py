"""Per-position distillation losses over the teacher's top-K candidates.

All functions take
    s_logits: student logits at the candidate tokens, shape (N, K), in the teacher's order (index 0 = teacher's top-1)
and return a per-position loss of shape (N,). The order-only losses (paper Table 2) never see teacher values.
OPD (paper Eq. 2) also needs the teacher's log-probabilities at the same K tokens.
"""
import torch
import torch.nn.functional as F


def adjacent_logistic(s_logits):
    """O3PD default: mean over i of -log sigmoid(z_i - z_{i+1}) (paper Eqs. 3-4)."""
    d = s_logits[:, :-1] - s_logits[:, 1:]
    return F.softplus(-d).mean(-1)


def _pair_mask(K, device):
    i = torch.arange(K, device=device)
    return i[:, None] < i[None, :]  # (i, j) with i ranked above j


def all_pairs_logistic(s_logits, weighted=False):
    """Logistic over all K(K-1)/2 ordered pairs; weighted=True gives Top-Weighted Logistic (w_ij = 1/i, 1-based)."""
    K = s_logits.shape[-1]
    d = s_logits[:, :, None] - s_logits[:, None, :]  # d[n, i, j] = z_i - z_j
    m = _pair_mask(K, s_logits.device).float()
    if weighted:
        m = m / torch.arange(1, K + 1, device=s_logits.device, dtype=s_logits.dtype)[:, None]
    return (F.softplus(-d) * m).sum((-1, -2)) / m.sum()


def listmle(s_logits):
    """Plackett-Luce negative log-likelihood of the teacher order, averaged over the K-1 choices."""
    # log sum_{j >= i} exp(z_j), computed right to left
    rev_lse = torch.logcumsumexp(s_logits.flip(-1), -1).flip(-1)
    return -(s_logits - rev_lse)[:, :-1].mean(-1)


def topk_jsd(s_logits, t_logprobs, temperature=1.0):
    """OPD baseline: JSD between teacher and student distributions restricted to the teacher's top-K and
    renormalized. Teacher logits are divided by `temperature` (the student's are not)."""
    p = torch.softmax(t_logprobs / temperature, -1)
    q = torch.softmax(s_logits, -1)
    m = 0.5 * (p + q)
    logm = torch.log(m.clamp_min(1e-30))
    kl_pm = (p * (torch.log(p.clamp_min(1e-30)) - logm)).sum(-1)
    kl_qm = (q * (torch.log_softmax(s_logits, -1) - logm)).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


LOSSES = {
    "adjacent": lambda s, t, T: adjacent_logistic(s),
    "logistic": lambda s, t, T: all_pairs_logistic(s),
    "topweighted": lambda s, t, T: all_pairs_logistic(s, weighted=True),
    "listmle": lambda s, t, T: listmle(s),
    "opd": lambda s, t, T: topk_jsd(s, t, T),
}


@torch.no_grad()
def agreement_metrics(s_full_logits, t_idx, t_logprobs, k=8):
    """Figure 5 metrics per position, averaged: JSD@top-k, Kendall tau on teacher's top-k, top-k overlap,
    top-1 agreement, student entropy (full vocabulary).

    s_full_logits: (N, V) student logits; t_idx: (N, K) teacher top-K token ids in order; t_logprobs: (N, K).
    """
    s_full_logits = s_full_logits.float()
    s_at = s_full_logits.gather(-1, t_idx[:, :k])
    jsd = topk_jsd(s_at, t_logprobs[:, :k].float()).mean().item()
    d_s = torch.sign(s_at[:, :, None] - s_at[:, None, :])
    m = _pair_mask(k, s_at.device)
    # teacher order is descending by construction, so teacher sign is +1 on every (i<j) pair
    tau = (d_s * m).sum((-1, -2)) / m.sum()
    s_top = s_full_logits.topk(k, -1).indices
    overlap = (s_top[:, :, None] == t_idx[:, None, :k]).any(-1).float().mean(-1)
    top1 = (s_top[:, 0] == t_idx[:, 0]).float()
    logp = torch.log_softmax(s_full_logits, -1)
    ent = -(logp.exp() * logp).sum(-1)
    return {"jsd_top8": jsd, "kendall_tau": tau.mean().item(), "top8_overlap": overlap.mean().item(),
            "top1_agree": top1.mean().item(), "student_entropy": ent.mean().item()}
