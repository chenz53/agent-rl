# SFT and preference optimization establish different baselines

Having understood outcome-based updates, compare two other signals on the same issue: a demonstrated correct clamp patch, and a pair consisting of a general bounds fix versus an inside-only patch. We can teach the demonstrated tokens or learn a preference between whole repair traces. Neither signal is identical to executing independent tests.

Agentic RL needs a competent starting policy and meaningful comparisons. Supervised trajectories can teach formatting and tool use; preference pairs can favor one complete behavior over another. These objectives belong in a broader post-training tutorial because they often solve problems that an outcome-only RL experiment cannot identify.

## SFT is selected-token likelihood with a supplied target

For demonstrated assistant tokens, SFT minimizes negative selected-token log-probability. The selection now follows the target sequence rather than a freshly sampled policy trajectory. Prompt and tool records remain conditioning input. A trace can teach correct formatting without demonstrating robust evidence selection; measure these separately.

## DPO compares two response likelihood ratios

For chosen and rejected responses, define the difference between their policy-versus-reference log-likelihood ratios. DPO applies a logistic loss to beta times that margin. The reference is fixed in this objective. Positive margins favor the labeled winner, but pair construction determines what behavior is learned.

The following two artificial pairs verify the sign of the loss. The numbers are sequence log-probabilities, not measured coding-agent responses.

## Construct the DPO loss

Subtract rejected sequence log-likelihood from chosen likelihood for both current and reference policies. Subtract the reference margin from the current margin and multiply by beta. `logsigmoid` evaluates the logistic preference objective stably; negation makes it a minimization loss. `mean` averages the example pairs. Reference terms are detached. The function is defined below, with no project helper import.

```python
import torch
def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = beta * ((chosen-rejected) - (ref_chosen-ref_rejected).detach())
    return -torch.nn.functional.logsigmoid(margin).mean()
ref_chosen = torch.tensor([-4.0])
ref_rejected = torch.tensor([-4.0])
for chosen, rejected in [(-3.0, -5.0), (-5.0, -3.0)]:
    loss = dpo_loss(torch.tensor([chosen]), torch.tensor([rejected]), ref_chosen, ref_rejected, beta=0.5)
    print(f"chosen={chosen:.1f}, rejected={rejected:.1f}, loss={loss.item():.6f}")
```

Output:

```text
chosen=-3.0, rejected=-5.0, loss=0.313262
chosen=-5.0, rejected=-3.0, loss=1.313262
```

## Design problems

If the preferred answer is longer, more confident, and better supported simultaneously, which feature drives the preference? Construct counterbalanced pairs and inspect length sensitivity. A preference for claiming that tests passed does not establish execution. Check the claim against the captured test command, exit code, and repository revision.

Offline pairs also come from a particular generator distribution. Online sampling can reveal new failures, but adds cost and evaluator instability. Keep the held-out set out of both preference construction and reward tuning. For coding tasks, preferring a teacher's patch does not establish that it passes independent regression tests.

The original objective is in [Direct Preference Optimization](https://arxiv.org/abs/2305.18290). This example does not include data collection, SFT training, or a deployed preference model.

**Exercise.** Create a pair differing only in whether a fabricated test result is claimed. Then create a pair differing only in unnecessary tool calls. Explain which evaluator can label each pair reliably.

**Next.** Study teacher guidance at student-visited prefixes in [on-policy distillation](10-on-policy-distillation.md).
