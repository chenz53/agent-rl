# Early tutor responses: English rendering

This is an English rendering of the two early teaching responses, not verbatim English conversation. The original Chinese text is preserved in Git history. The initial proposals evolved after user feedback; the current application is oncology prognosis.

## Response 1: initial teaching approach

The proposed sequence was intuition, mechanism, implementation, failure analysis, and critical paper reading. The goal was independent training design, diagnosis of failures, and judgment about research reliability.

Agentic RL was introduced as training an LLM that makes multiple decisions, uses tools, and interacts with an environment. The initial roadmap covered:

| Area | Question | Practical artifact |
| --- | --- | --- |
| RL foundations | What are states, actions, rewards, and delayed consequences? | A hand-computable environment |
| Optimization | Why do policy gradients work? What do values, advantages, and exploration solve? | REINFORCE and PPO examples |
| LLM RL | How do tokens, SFT, KL, and GRPO connect? | A sampling-to-update walkthrough |
| Multi-step agents | How does feedback change decisions and receive credit? | A tool-using agent |
| Systems | How do reset, parallelism, timeouts, policy lag, and cost interact? | Reproducible training/evaluation |
| Research diagnosis | Why can reward rise without capability gains? | Ablations and failure analysis |

SCoRe and Agent Lightning were candidate readings. The first concrete example was a coding agent: inspect code, edit it, run tests, observe errors, revise, and submit. Observations are code/tool results; actions are operations; the policy is a conditional action distribution; the trajectory is the interaction; reward is training feedback.

The expected cumulative reward objective was:

\[
J(\theta)=E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T}\gamma^t r_t\right].
\]

The credit-assignment question was which earlier actions should be encouraged after final success or failure. A separate question was whether passing tests really established a correct fix. The tutor asked about programming, mathematical background, preferred agent domain, and the difficulty of learning from only a terminal 0/1 reward.

## Response 2: adapt to foundation-model expertise

After learning the user's experience, the tutor raised the starting level: transfer model-training expertise to sequential decision optimization rather than repeat deep-learning foundations. Each biomedical example should connect mathematics, implementation, failure modes, and distinguishing experiments.

The revised topics were environment design, multi-turn policy objectives, credit assignment, informative exploration, systems coordination, and scientific validity. The starting question became what is being optimized, before choosing PPO or GRPO.

An example task asked an agent to evaluate gene-disease evidence up to a cutoff. Three capabilities were separated: existing knowledge, information acquisition, and evidence judgment. Otherwise memorization could be mistaken for research skill.

For task x and tool environment E:

\[
p_\theta(\tau\mid x,E)=p(o_0\mid x,E)
\prod_t\pi_\theta(a_t\mid h_t)P_E(o_{t+1}\mid h_t,a_t).
\]

This factorization explains why a changed policy changes later inputs, why database/retrieval versions and context truncation belong in the training definition, and why tools need not be differentiable. Their outputs influence later behavior and reward, while generated actions contribute policy likelihood terms.

Finally, the tutor distinguished computable reward R from true scientific utility U. A citation existing, supporting a statement, and establishing causality are different standards. Optimizing the first need not improve the last. ATHENA-R1 was named as a candidate biomedical tool-training study, pending closer examination. The user was asked to choose among evidence synthesis, omics, target/drug discovery, and clinical reasoning; they subsequently chose clinical reasoning.
