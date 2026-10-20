"""Does the cascade actually pay for itself? Compute the break-even acceptance rate.

The question this answers, in one sentence: *at what first-tier approval rate does
the cascade cost the same as sending every request straight to the final tier?*

The arithmetic, per request, with ``p`` the probability that level 1 is approved:

    P(both L1 and L2 rejected) = (1-p)(1-q)      # q = L2 acceptance rate
    expected_cost(p) = c1 + (1-p)·c2 + (1-p)(1-q)·c3
    baseline_cost    = c3

where ``ci`` is the cost of one call to level ``i``. Note that the last tier is
paid for *in addition* to the rejected candidates: a cascade is not free when it
works, it is free when it works **early**.

``evaluate`` finds the ``p`` where ``expected_cost(p) == baseline``. If even a
0% approval rate is cheaper than the baseline, no break-even exists; if even a
100% rate is dearer, the cascade never pays for itself.

Costs are supplied by the caller — this module never invents a price.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BreakevenAnalysis:
    """The result of one break-even computation."""

    #: Cost of one call at each level, from level 1 to the final level.
    stage_costs: tuple[float, ...]
    #: Acceptance rate assumed for every evaluated level except the last.
    secondary_acceptance: float
    #: Cost of answering every request directly with the final level.
    baseline_cost: float
    #: Minimum level-1 acceptance rate at which the cascade breaks even, or None.
    breakeven_acceptance: float | None
    #: Expected cascade cost at 0% and at 100% first-tier acceptance.
    cost_at_zero_acceptance: float
    cost_at_full_acceptance: float

    @property
    def always_cheaper(self) -> bool:
        """Even with nothing approved, the cascade costs less than the baseline."""
        return self.cost_at_zero_acceptance <= self.baseline_cost

    @property
    def never_cheaper(self) -> bool:
        """Even with everything approved, the cascade costs more than the baseline."""
        return self.cost_at_full_acceptance > self.baseline_cost

    def expected_cost(self, acceptance: float) -> float:
        """Expected cost per request at a given level-1 acceptance rate."""
        remaining = 1.0 - acceptance
        total = self.stage_costs[0]
        for cost in self.stage_costs[1:]:
            # A level is paid for whenever the request reaches it, rejected
            # candidates included: that is precisely why a cascade is not free.
            total += remaining * cost
            remaining *= 1.0 - self.secondary_acceptance
        return total

    def saving_at(self, acceptance: float) -> float:
        """Expected USD saved per request (negative means the cascade costs more)."""
        return self.baseline_cost - self.expected_cost(acceptance)


def evaluate(stage_costs, *, secondary_acceptance: float = 0.8) -> BreakevenAnalysis:
    """Compute the break-even first-tier acceptance rate.

    ``secondary_acceptance`` applies to every evaluated level but the last; with a
    three-tier cascade that is the level-2 rate. It is an assumption, so it is an
    explicit argument with a documented default rather than a hidden constant.
    """
    costs = tuple(float(cost) for cost in stage_costs)
    if len(costs) < 2:
        raise ValueError("A break-even analysis needs at least two cascade levels")
    if any(cost < 0 for cost in costs):
        raise ValueError(f"Stage costs cannot be negative: {costs}")
    if not 0.0 <= secondary_acceptance <= 1.0:
        raise ValueError(
            f"secondary_acceptance must be between 0 and 1, got {secondary_acceptance!r}"
        )

    baseline = costs[-1]
    analysis = BreakevenAnalysis(
        stage_costs=costs,
        secondary_acceptance=secondary_acceptance,
        baseline_cost=baseline,
        breakeven_acceptance=None,
        cost_at_zero_acceptance=0.0,
        cost_at_full_acceptance=0.0,
    )
    zero = analysis.expected_cost(0.0)
    full = analysis.expected_cost(1.0)

    breakeven: float | None = None
    if full <= baseline < zero:
        # Cost is linear in `p`: cost(p) = full + u(cost(0) - full) with u = 1 - p.
        # Solve cost(p) = baseline for u, then convert back to an acceptance rate.
        unmet = (baseline - full) / (zero - full)
        breakeven = min(1.0, max(0.0, 1.0 - unmet))

    return BreakevenAnalysis(
        stage_costs=costs,
        secondary_acceptance=secondary_acceptance,
        baseline_cost=baseline,
        breakeven_acceptance=breakeven,
        cost_at_zero_acceptance=zero,
        cost_at_full_acceptance=full,
    )


def costs_from_config(config, *, prompt_tokens: int, completion_tokens: int) -> tuple[float, ...]:
    """Cost of one average-sized call at each level, using the configured prices.

    Costs come from ``config`` (i.e. from `cascade.yaml` or `pricing.yaml`); this
    function only multiplies, so it can never invent a price.
    """
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("Token counts cannot be negative")
    return tuple(
        prompt_tokens * level.pricing.input_per_mtok_usd / 1_000_000
        + completion_tokens * level.pricing.output_per_mtok_usd / 1_000_000
        for level in config.levels
    )


def render_report(analysis: BreakevenAnalysis) -> str:
    lines = [
        "Break-even analysis",
        "===================",
        f"Stage costs: {', '.join(f'L{i+1}=${cost:.6f}' for i, cost in enumerate(analysis.stage_costs))}",
        f"Baseline (one call to the final level): ${analysis.baseline_cost:.6f}",
        f"Assumed acceptance at intermediate evaluated levels: {analysis.secondary_acceptance:.0%}",
        "",
        f"  acceptance  0%  ->  ${analysis.cost_at_zero_acceptance:.6f} per request",
        f"  acceptance 100% ->  ${analysis.cost_at_full_acceptance:.6f} per request",
        "",
    ]
    if analysis.always_cheaper:
        lines.append(
            "Verdict: the cascade is cheaper even if level 1 never has an answer "
            "approved. It pays for itself unconditionally."
        )
    elif analysis.never_cheaper:
        lines.append(
            "Verdict: even with every level-1 answer approved, the cascade costs more "
            "than going straight to the final level. It never pays for itself."
        )
    elif analysis.breakeven_acceptance is not None:
        lines.append(
            f"Verdict: the cascade breaks even above a "
            f"{analysis.breakeven_acceptance:.1%} level-1 approval rate. "
            "Below that, sending everything to the final level is cheaper."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="breakeven",
        description=(
            "Compute the first-tier acceptance rate at which the cascade costs the "
            "same as going straight to the final level."
        ),
    )
    parser.add_argument(
        "stage_costs",
        nargs="*",
        type=float,
        help="Average cost of ONE call at each level, cheapest first (USD).",
    )
    parser.add_argument(
        "--secondary-acceptance",
        type=float,
        default=0.8,
        help="Assumed acceptance rate at intermediate evaluated levels (default 0.8).",
    )
    parser.add_argument(
        "--from-config",
        metavar="PATH",
        help="Derive the stage costs from a cascade config file (uses its prices).",
    )
    parser.add_argument("--prompt-tokens", type=int, default=1000,
                        help="Average prompt tokens per call (default 1000).")
    parser.add_argument("--completion-tokens", type=int, default=500,
                        help="Average completion tokens per call (default 500).")
    args = parser.parse_args(argv)

    stage_costs = args.stage_costs
    if not args.from_config and len(stage_costs) < 2:
        print(
            "breakeven failed: give at least two stage costs, "
            "or use --from-config PATH to read them from a cascade config.",
            file=sys.stderr,
        )
        return 2
    if args.from_config:
        # Imported lazily: the harness reads real files only when asked to.
        from src.cascade_config import load_cascade_config

        try:
            config = load_cascade_config(Path(args.from_config), log_warnings=False)
            stage_costs = list(costs_from_config(
                config,
                prompt_tokens=args.prompt_tokens,
                completion_tokens=args.completion_tokens,
            ))
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(f"breakeven failed: {exc}", file=sys.stderr)
            return 2

    try:
        analysis = evaluate(stage_costs, secondary_acceptance=args.secondary_acceptance)
    except ValueError as exc:
        print(f"breakeven failed: {exc}", file=sys.stderr)
        return 2

    print(render_report(analysis))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
