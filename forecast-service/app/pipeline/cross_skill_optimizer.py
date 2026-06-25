"""Cross-skill rebalancing optimiser (min-cost transportation).

The default borrow/lend recommendation is a greedy fill (largest donor → largest
receiver). It maximizes total FTE moved but is blind to *where* the lend comes
from, so it will move staff across unrelated business areas as readily as within
one. This optimiser solves the same coverage problem as a min-cost max-flow: it
moves at least as much effective FTE as the greedy (same maximum coverage) while
minimizing a transfer-affinity cost, so lends prefer the same sub-BA, then the
same BA, then the same channel, before crossing the org. Pure Python — no solver
dependency — and provably optimal for the transportation structure.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Optional

# Default affinity costs (lower = more preferred). Same team is free.
DEFAULT_COSTS = {"same_sba": 0, "same_ba": 1, "same_channel": 2, "cross": 4}
_SCALE = 100  # work in integer centi-FTE


def _norm(v: Any) -> str:
    return str(v or "").strip().lower()


def _affinity_cost(donor: dict, recv: dict, costs: dict) -> int:
    d_ba, r_ba = _norm(donor.get("ba")), _norm(recv.get("ba"))
    d_sba, r_sba = _norm(donor.get("sba")), _norm(recv.get("sba"))
    d_ch, r_ch = _norm(donor.get("ch")), _norm(recv.get("ch"))
    if d_ba and d_ba == r_ba and d_sba and d_sba == r_sba:
        return int(costs.get("same_sba", 0))
    if d_ba and d_ba == r_ba:
        return int(costs.get("same_ba", 1))
    if d_ch and d_ch == r_ch:
        return int(costs.get("same_channel", 2))
    return int(costs.get("cross", 4))


class _MCMF:
    """Min-cost max-flow via SPFA (Bellman-Ford) successive shortest paths."""

    def __init__(self, n: int):
        self.n = n
        self.g: list[list[list]] = [[] for _ in range(n)]  # edge: [to, cap, cost, rev_idx]

    def add_edge(self, u: int, v: int, cap: int, cost: int) -> None:
        self.g[u].append([v, cap, cost, len(self.g[v])])
        self.g[v].append([u, 0, -cost, len(self.g[u]) - 1])

    def run(self, s: int, t: int) -> tuple[int, int]:
        INF = float("inf")
        flow = 0
        cost = 0
        while True:
            dist = [INF] * self.n
            in_q = [False] * self.n
            prevv = [-1] * self.n
            preve = [-1] * self.n
            dist[s] = 0
            q = deque([s])
            in_q[s] = True
            while q:
                u = q.popleft()
                in_q[u] = False
                du = dist[u]
                for i, e in enumerate(self.g[u]):
                    v, cap, ecost, _ = e
                    if cap > 0 and du + ecost < dist[v]:
                        dist[v] = du + ecost
                        prevv[v] = u
                        preve[v] = i
                        if not in_q[v]:
                            q.append(v)
                            in_q[v] = True
            if dist[t] == INF:
                break
            # bottleneck along the path
            d = INF
            v = t
            while v != s:
                d = min(d, self.g[prevv[v]][preve[v]][1])
                v = prevv[v]
            v = t
            while v != s:
                e = self.g[prevv[v]][preve[v]]
                e[1] -= d
                self.g[v][e[3]][1] += d
                v = prevv[v]
            flow += d
            cost += d * dist[t]
        return flow, cost


def optimize(
    scope_balance: list[dict],
    xskill_eff: float,
    max_lend_ratio: float,
    costs: Optional[dict] = None,
    max_moves: int = 50,
) -> dict:
    """Return optimised cross-skill moves + a summary, from a scope-balance table."""
    costs = {**DEFAULT_COSTS, **(costs or {})}
    eff = max(1e-6, float(xskill_eff or 0.0))
    lend_ratio = max(0.0, float(max_lend_ratio or 0.0))

    donors = []
    for s in scope_balance or []:
        surplus = float(s.get("surplus_fte") or 0.0)
        if surplus > 0 and not bool(s.get("locked_critical")):
            lendable = surplus * lend_ratio
            if lendable > 0:
                donors.append({**s, "_lendable": lendable})
    receivers = []
    for s in scope_balance or []:
        short = float(s.get("shortfall_fte") or 0.0)
        if short > 0:
            receivers.append({**s, "_need_gross": short / eff, "_short": short})

    if not donors or not receivers:
        return {
            "moves": [],
            "summary": {
                "total_lend_fte": 0.0,
                "total_effective_fte": 0.0,
                "post_rebalance_shortfall_fte": round(sum(r["_short"] for r in receivers), 2),
                "within_ba_pct": None,
                "within_channel_pct": None,
                "donors": len(donors),
                "receivers": len(receivers),
            },
        }

    D, R = len(donors), len(receivers)
    S, T = 0, D + R + 1  # source, sink
    mcmf = _MCMF(D + R + 2)
    for i, d in enumerate(donors):
        mcmf.add_edge(S, 1 + i, int(round(d["_lendable"] * _SCALE)), 0)
    for j, r in enumerate(receivers):
        mcmf.add_edge(1 + D + j, T, int(round(r["_need_gross"] * _SCALE)), 0)
    for i, d in enumerate(donors):
        for j, r in enumerate(receivers):
            mcmf.add_edge(1 + i, 1 + D + j, 10 ** 9, _affinity_cost(d, r, costs))

    mcmf.run(S, T)

    # Decode donor→receiver flows (flow on a forward edge = its reverse edge's cap).
    moves = []
    for i, d in enumerate(donors):
        for e in mcmf.g[1 + i]:
            v, _cap, ecost, rev = e
            if 1 + D <= v <= D + R:  # edge into a receiver node
                flow_units = mcmf.g[v][rev][1]
                if flow_units > 0:
                    j = v - (1 + D)
                    r = receivers[j]
                    lend = flow_units / _SCALE
                    moves.append(
                        {
                            "from_scope": str(d.get("scope")),
                            "to_scope": str(r.get("scope")),
                            "lend_fte": round(lend, 2),
                            "effective_fte": round(lend * eff, 2),
                            "affinity": ecost,
                            "same_ba": _norm(d.get("ba")) == _norm(r.get("ba")) and bool(_norm(d.get("ba"))),
                            "same_channel": _norm(d.get("ch")) == _norm(r.get("ch")) and bool(_norm(d.get("ch"))),
                        }
                    )
    moves.sort(key=lambda m: (m["affinity"], -m["effective_fte"]))
    if max_moves and len(moves) > max_moves:
        moves = moves[:max_moves]

    total_lend = sum(m["lend_fte"] for m in moves)
    total_eff = sum(m["effective_fte"] for m in moves)
    total_short = sum(r["_short"] for r in receivers)
    within_ba = sum(m["effective_fte"] for m in moves if m["same_ba"])
    within_ch = sum(m["effective_fte"] for m in moves if m["same_channel"])
    return {
        "moves": moves,
        "summary": {
            "total_lend_fte": round(total_lend, 2),
            "total_effective_fte": round(total_eff, 2),
            "post_rebalance_shortfall_fte": round(max(0.0, total_short - total_eff), 2),
            "within_ba_pct": round(within_ba / total_eff * 100.0, 1) if total_eff > 0 else None,
            "within_channel_pct": round(within_ch / total_eff * 100.0, 1) if total_eff > 0 else None,
            "donors": D,
            "receivers": R,
        },
    }
