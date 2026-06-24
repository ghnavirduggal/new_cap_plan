"""Monte Carlo demand simulation for capacity risk.

Required FTE responds to demand volume through the (nonlinear) capacity engine, so
running thousands of full capacity computes is infeasible. Instead the endpoint
builds a small required-FTE *response curve* over a handful of volume multipliers;
this module draws many demand multipliers from a lognormal distribution (mean 1,
given CV), interpolates required FTE from the curve for each draw, and summarizes
the distribution plus the probability that supply covers demand.
"""
from __future__ import annotations

import math
import random
from typing import Optional


def lognormal_params(cv: float) -> tuple[float, float]:
    """Underlying-normal (mu, sigma) for a lognormal with mean 1 and the given CV."""
    cv = max(float(cv), 1e-6)
    sigma = math.sqrt(math.log(1.0 + cv * cv))
    mu = -0.5 * sigma * sigma  # so E[exp(N(mu,sigma))] = 1
    return mu, sigma


def interp(curve: list[tuple[float, float]], x: float) -> float:
    """Linear interpolation of required FTE at multiplier x. The curve is a sorted
    list of (multiplier, required). x is clamped to the curve's range so rare tail
    draws don't extrapolate beyond what was actually computed."""
    if not curve:
        return 0.0
    if x <= curve[0][0]:
        return curve[0][1]
    if x >= curve[-1][0]:
        return curve[-1][1]
    for i in range(1, len(curve)):
        x0, y0 = curve[i - 1]
        x1, y1 = curve[i]
        if x <= x1:
            if x1 == x0:
                return y1
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return curve[-1][1]


def simulate(
    curve: list[tuple[float, float]],
    supply: Optional[float],
    cv: float,
    draws: int = 2000,
    seed: int = 12345,
) -> dict:
    draws = max(100, min(int(draws or 2000), 50000))
    rng = random.Random(seed)
    mu, sigma = lognormal_params(cv)
    reqs: list[float] = []
    covered = 0
    for _ in range(draws):
        m = math.exp(rng.gauss(mu, sigma))
        r = interp(curve, m)
        reqs.append(r)
        if supply is not None and supply >= r:
            covered += 1
    reqs.sort()

    def pct(p: float) -> float:
        if not reqs:
            return 0.0
        idx = min(len(reqs) - 1, max(0, int(round(p * (len(reqs) - 1)))))
        return round(reqs[idx], 1)

    coverage = (covered / draws) if (supply is not None and draws) else None
    return {
        "draws": draws,
        "cv": round(float(cv), 4),
        "supply": None if supply is None else round(supply, 1),
        "required_p10": pct(0.10),
        "required_p50": pct(0.50),
        "required_p90": pct(0.90),
        "required_p95": pct(0.95),
        "required_mean": round(sum(reqs) / len(reqs), 1) if reqs else 0.0,
        "coverage_prob": None if coverage is None else round(coverage, 4),
        "shortfall_prob": None if coverage is None else round(1.0 - coverage, 4),
    }
