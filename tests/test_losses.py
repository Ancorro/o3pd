import torch

from o3pd.losses import LOSSES, adjacent_logistic, agreement_metrics, all_pairs_logistic, listmle, topk_jsd

torch.manual_seed(0)


def teacher_topk(t_logits, k=8, T=1.0):
    lp, idx = torch.log_softmax(t_logits / T, -1).topk(k, -1)
    return lp, idx


def test_order_only_losses_ignore_teacher_temperature():
    s = torch.randn(16, 100)
    t = torch.randn(16, 100) * 3
    for name in ["adjacent", "logistic", "topweighted", "listmle"]:
        vals = []
        for T in [0.25, 1.0, 8.0]:
            lp, idx = teacher_topk(t, T=T)
            vals.append(LOSSES[name](s.gather(-1, idx), lp, T))
        assert torch.allclose(vals[0], vals[1]) and torch.allclose(vals[1], vals[2]), name


def test_opd_depends_on_teacher_temperature():
    s = torch.randn(16, 100)
    t = torch.randn(16, 100) * 3
    lp, idx = teacher_topk(t)
    a = topk_jsd(s.gather(-1, idx), lp, 1.0)
    b = topk_jsd(s.gather(-1, idx), lp, 4.0)
    assert not torch.allclose(a, b)


def test_adjacent_gradient_is_minus_sigmoid_of_minus_delta():
    z = torch.tensor([[2.0, 3.0, -1.0]], requires_grad=True)
    adjacent_logistic(z).sum().backward()
    d = torch.tensor([2.0 - 3.0, 3.0 - (-1.0)])
    g = -torch.sigmoid(-d) / 2  # mean over K-1 = 2 pairs
    expect = torch.tensor([[g[0], -g[0] + g[1], -g[1]]])
    assert torch.allclose(z.grad, expect, atol=1e-6)


def test_order_losses_small_when_order_matches_with_margin():
    z = torch.tensor([[40.0, 30.0, 20.0, 10.0]])
    for f in [adjacent_logistic, all_pairs_logistic, listmle]:
        assert f(z).item() < 1e-3
    assert adjacent_logistic(z.flip(-1)).item() > 5


def test_order_losses_shift_and_scale_free_targets():
    """Order-only: rescaling the *student* logits up keeps a correct order's loss going to 0 (no target margin)."""
    z = torch.tensor([[1.0, 0.5, 0.0]])
    assert adjacent_logistic(10 * z) < adjacent_logistic(z)


def test_jsd_zero_iff_match():
    lp = torch.log_softmax(torch.randn(4, 8), -1)
    assert topk_jsd(lp.clone(), lp).abs().max() < 1e-6
    assert topk_jsd(lp + torch.randn(4, 8), lp).min() > 0


def test_agreement_metrics_perfect_student():
    t = torch.randn(10, 50)
    lp, idx = teacher_topk(t)
    m = agreement_metrics(t, idx, lp)
    assert abs(m["kendall_tau"] - 1) < 1e-6 and m["top8_overlap"] == 1 and m["top1_agree"] == 1
    assert m["jsd_top8"] < 1e-6
