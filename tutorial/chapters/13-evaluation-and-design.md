# Evaluate the patch, not the agent's claim

A coding agent can claim “all tests pass” without running them, or pass one public test with an overfit patch. Chapter 2 separated public checks from evaluator checks. We now ask whether the reward and held-out measurement still agree with the intended repair task.

## Make failure modes observable

Track issue-test success, regression success, evaluator pass rate, patch validity, tool execution errors, resource use, and completion status separately. A successful tool invocation means the command ran; it does not establish that the bug was fixed. A positive reward for every successful test command can encourage repeated testing without better patches.

An inside-only implementation returns 3 for the one public case. It ignores lower and upper bounds. Comparing it with a general repair makes the evaluator gap concrete. The helper below runs all authored trusted candidates in fresh namespaces; it is reconstructed here rather than imported.

```python
def run_cases(source, cases):
    namespace = {}
    try:
        exec(source, namespace)
        function = namespace["clamp"]
    except Exception:
        return [False] * len(cases)
    results = []
    for arguments, expected in cases:
        try:
            actual = function(*arguments)
        except Exception as error:
            passed = isinstance(expected, type) and isinstance(error, expected)
        else:
            passed = not isinstance(expected, type) and actual == expected
        results.append(passed)
    return results

public_cases = [((3, 0, 5), 3)]
evaluator_cases = public_cases + [((-2, 0, 5), 0), ((8, 0, 5), 5), ((3, 5, 0), ValueError)]
patches = {
    "general repair": "def clamp(v, lo, hi):\n    if lo > hi:\n        raise ValueError()\n    return max(lo, min(v, hi))\n",
    "inside-only patch": "def clamp(v, lo, hi):\n    return 3\n",
    "unchanged input": "def clamp(v, lo, hi):\n    return v\n",
}
for name, source in patches.items():
    public_pass = all(run_cases(source, public_cases))
    evaluator_pass = all(run_cases(source, evaluator_cases))
    print(name, "public:", public_pass, "evaluator:", evaluator_pass)
```

Output:

```text
general repair public: True evaluator: True
inside-only patch public: True evaluator: False
unchanged input public: True evaluator: False
```

Public success does not distinguish these candidates, but the independent cases do. This is an actual execution result over authored source, not a generated-agent benchmark. Tests still cover only their stated domain; a passing patch can have untested defects.

## Access and test integrity

Prevent the candidate from overwriting evaluator tests, reference patches, or score files. Test modifications are not necessarily malicious—some issues require adding tests—but scoring should verify the application code against an independent trusted test set. Pin the starting repository, dependencies, and test command. Separate a failed patch from an infrastructure failure or timeout; silently turning all execution problems into zero can distort learning.

## Compare at matched budgets

One-sample solve rate differs from pass@G, which asks whether any of G attempts succeeds. A deployable selected candidate requires a selection strategy: the best candidate under the private evaluator is an oracle measurement, not an available production selector. More attempts can improve success without any parameter update. Match generated tokens, tool calls, candidate count, and teacher cost when comparing training methods.

Estimate uncertainty across independent coding issues rather than treating several rollouts of the same issue as independent tasks. Separate training tests, validation tasks used to tune recipes, and final held-out repositories. Track cost alongside correctness before combining them into one reward; a large step penalty can favor premature submission.

**Exercise.** Design a patch that alters a public test to pass without changing `clamp`. What should the verifier mount read-only? Then compare a direct patch-generation baseline with a multi-step repair agent at equal inference budget.

**Next.** Read [research alternatives](14-research-frontier.md), then execute [the coding capstone](15-coding-capstone.md).
