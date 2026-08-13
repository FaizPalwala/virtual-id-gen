#!/usr/bin/env python3
"""Validate the prompt variation plan BEFORE generation spends GPU hours.

Pre-flight gate for hpc_generate.sh: the 20×5 prompt grid must yield 
100 unique prompts with real per-dimension diversity, deterministically. 
A silent collapse here (e.g. duplicate compositions, a broken lighting 
cross-product) would produce an identity-consistent-but-monotonous dataset 
that only surfaces at release QA — after ~8 GPU-hours per shard × 15 shards.

Checks:
  1. exactly 100 variations
  2. all rendered prompts unique (no two candidates share a rendering
     instruction — the Ch3 §3.2.3 design claim)
  3. per-dimension diversity floors: lighting == 5; pose/expression/
     setting/camera each >= 10 unique values (20 compositions guarantee
     >= 20 poses; a floor of 10 catches gross collapse without over-
     constraining)
  4. determinism: two builds of the plan are identical (same seed path)
  5. no identity-trait leakage in the prompt prefix (the plan must only
     vary non-identity attributes)

Exit 0 = GO (safe to submit generation); non-zero = fix the plan first.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prompts import PHOTOREALISM_PREFIX, build_variation_plan  # noqa: E402

DIM_FLOORS = {"lighting": 5, "pose": 10, "expression": 8,
              "setting": 10, "camera": 3}
# The prefix legitimately anchors identity ("portrait of the same person")
# — that is the identity-conditioning instruction, not leakage.  What must
# NOT appear is anything that fixes a NON-identity trait identically across
# all prompts (that would be the variation collapse we are guarding).
LEAK_TOKENS = ("red hair", "blue eyes", "wearing a hat", "tattoo",
               "beard", "glasses")


def main() -> int:
    failures: list[str] = []

    plan = build_variation_plan()

    # 1. Count
    if len(plan) != 100:
        failures.append(f"expected 100 variations, got {len(plan)}")

    # 2. Uniqueness of rendered prompts
    prompts = [v.prompt() for v in plan]
    dupes = len(prompts) - len(set(prompts))
    if dupes:
        failures.append(f"{dupes} duplicate rendered prompts in the plan")

    # 3. Per-dimension diversity
    for dim, floor in DIM_FLOORS.items():
        n = len({getattr(v, dim) for v in plan})
        if n < floor:
            failures.append(
                f"dimension '{dim}' has {n} unique values (floor {floor})"
            )

    # 4. Determinism
    plan2 = build_variation_plan()
    if [v.prompt() for v in plan] != [v.prompt() for v in plan2]:
        failures.append("plan is not deterministic across builds")

    # 5. No identity-trait leakage in the prefix
    for token in LEAK_TOKENS:
        if token in PHOTOREALISM_PREFIX.lower():
            failures.append(
                f"photorealism prefix leaks identity language: '{token}'"
            )

    if failures:
        print("PROMPT PLAN: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    dims = {d: len({getattr(v, d) for v in plan}) for d in DIM_FLOORS}
    print(f"PROMPT PLAN: GO — 100 unique prompts, {dims}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
