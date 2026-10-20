"""Figure for Chapter 1: the agentic RL loop, drawn in the hairline style (scripts/hairline.py)."""
import hairline


def agentic_rl_loop():
    (hairline.OUT / 'agentic-rl-loop.svg').write_text(hairline.agent_loop())


FIGURES = {'agentic-rl-loop': agentic_rl_loop}
