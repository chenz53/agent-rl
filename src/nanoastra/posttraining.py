"""Small PyTorch teaching primitives, not a distributed training framework."""
import torch
import torch.nn.functional as F


def selected_log_probs(logits, token_ids):
    """Inputs [..., V], [...] -> [...]; vocabulary normalization remains full."""
    if logits.shape[:-1] != token_ids.shape:
        raise ValueError("Token IDs must match every non-vocabulary axis")
    return logits.log_softmax(-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


def group_advantages(rewards, normalize=False, leave_one_out=False):
    """[tasks, rollouts] -> same shape; explicitly select the estimator."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Need at least two rollouts per task")
    baseline = rewards.mean(-1, keepdim=True)
    if leave_one_out:
        baseline = (rewards.sum(-1, keepdim=True) - rewards) / (rewards.shape[1] - 1)
    advantage = rewards - baseline
    if normalize:
        advantage = advantage / rewards.std(-1, keepdim=True, correction=0).clamp_min(1e-8)
    return advantage.detach()


def reduce_tokens(values, mask, reduction="sequence_sum"):
    """Choose trajectory weighting explicitly; accepts [..., time]."""
    counts = mask.sum(-1)
    if (counts == 0).any():
        raise ValueError("Empty action sequence; diagnose collection rather than hiding it")
    totals = (values * mask).sum(-1)
    if reduction == "sequence_sum":
        return totals.mean()
    if reduction == "sequence_mean":
        return (totals / counts).mean()
    if reduction == "token_mean":
        return totals.sum() / counts.sum()
    raise ValueError(reduction)


def policy_gradient_loss(log_probs, advantage, mask, reduction="sequence_sum"):
    return -reduce_tokens(log_probs * advantage.detach().unsqueeze(-1), mask, reduction)


def clipped_policy_loss(log_probs, old_log_probs, advantage, mask, eps=0.2):
    """Token-ratio PPO surrogate only; no critic/KL/rollout correction included."""
    ratio = (log_probs - old_log_probs.detach()).exp()
    weights = advantage.detach().unsqueeze(-1)
    surrogate = torch.minimum(ratio * weights, ratio.clamp(1-eps, 1+eps) * weights)
    return -reduce_tokens(surrogate, mask, "token_mean")


def reverse_kl(student_logits, teacher_logits):
    """Exact KL(student || teacher) at fixed prefixes, output [...]."""
    logp = student_logits.log_softmax(-1)
    logq = teacher_logits.detach().log_softmax(-1)
    return (logp.exp() * (logp - logq)).sum(-1)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = beta * ((chosen-rejected) - (ref_chosen-ref_rejected).detach())
    return -F.logsigmoid(margin).mean()
