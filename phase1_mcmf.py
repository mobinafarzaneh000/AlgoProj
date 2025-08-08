#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 — Initial Task-to-Node Allocation using MCMF (Successive Shortest Path)
No external libraries. Uses only Python standard library.

Input JSON schema:
{
  "tasks":[{"id":"T1","cpu":2,"ram":4,"deadline":2}, ...],
  "nodes":[{"id":"N1","cpu_capacity":5,"ram_capacity":6,"task_limit":2}, ...],
  "exec_cost":{"T1":{"N1":4,"N2":6}, "T2":{"N1":3}}
}
Output JSON:
{
  "valid": true/false,
  "assignments": {"T1":"N1", ...},
  "total_cost": 9,
  "unassignable_tasks": ["T5", ...],     # (اگر وجود داشته باشد)
  "reason": "..."                         # (اگر invalid باشد)
}
"""

import json
import sys
import argparse
import heapq
from typing import Dict, List, Tuple, Optional

INF = 10**18


# ------------------------ Core MCMF (SSP + Potentials) ------------------------

class MinCostMaxFlow:
    """Minimum-Cost Max-Flow using Successive Shortest Path with Johnson potentials."""

    class Edge:
        __slots__ = ("u", "v", "cap", "cost", "rev")
        def __init__(self, u: int, v: int, cap: int, cost: int, rev: Optional["MinCostMaxFlow.Edge"]):
            self.u = u
            self.v = v
            self.cap = cap
            self.cost = cost
            self.rev = rev  # reverse edge

    def __init__(self, n: int):
        self.n = n
        self.g: List[List[MinCostMaxFlow.Edge]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int, cost: int):
        a = MinCostMaxFlow.Edge(u, v, cap, cost, None)
        b = MinCostMaxFlow.Edge(v, u, 0, -cost, None)
        a.rev = b
        b.rev = a
        self.g[u].append(a)
        self.g[v].append(b)

    def min_cost_flow(self, s: int, t: int, want_flow: int) -> Tuple[int, int]:
        """Return (flow_sent, total_cost) up to want_flow."""
        n = self.n
        flow = 0
        cost = 0
        pot = [0] * n  # vertex potentials; costs are assumed non-negative after reweighting

        # If initial negative edges existed, we could run Bellman-Ford once here.
        # In our model, costs are non-negative.

        while flow < want_flow:
            dist = [INF] * n
            parent: List[Optional[MinCostMaxFlow.Edge]] = [None] * n
            dist[s] = 0
            pq: List[Tuple[int, int]] = [(0, s)]

            while pq:
                d, u = heapq.heappop(pq)
                if d != dist[u]:
                    continue
                for e in self.g[u]:
                    if e.cap <= 0:
                        continue
                    nd = d + e.cost + pot[u] - pot[e.v]
                    if nd < dist[e.v]:
                        dist[e.v] = nd
                        parent[e.v] = e
                        heapq.heappush(pq, (nd, e.v))

            if parent[t] is None:
                break  # no augmenting path

            # update potentials
            for v in range(n):
                if dist[v] < INF:
                    pot[v] += dist[v]

            # bottleneck
            add = want_flow - flow
            v = t
            while v != s:
                e = parent[v]
                add = min(add, e.cap)
                v = e.u

            # push
            v = t
            while v != s:
                e = parent[v]
                e.cap -= add
                e.rev.cap += add
                cost += add * e.cost
                v = e.u

            flow += add

        return flow, cost


# --------------------------- Phase 1 Graph Builder ----------------------------

def _estimate_task_limit(tasks: List[dict], node: dict) -> int:
    """
    اگر task_limit مشخص نبود، یک سقف معقول برای تعداد کار روی گره برآورد می‌کند.
    این یک تقریبِ محتاطانه است (برای فاز ۱ کافی است).
    """
    if not tasks:
        return 0
    min_cpu_need = max(1, min(t["cpu"] for t in tasks))
    min_ram_need = max(1, min(t["ram"] for t in tasks))
    by_cpu = max(1, node["cpu_capacity"] // min_cpu_need)
    by_ram = max(1, node["ram_capacity"] // min_ram_need)
    return max(1, min(by_cpu, by_ram))


def build_phase1_graph(tasks: List[dict], nodes: List[dict],
                       exec_cost: Dict[str, Dict[str, int]],
                       fallback_task_limit: Optional[int] = None):
    """
    S -> Ti (cap=1,c=0)
    Ti -> Nj (cap=1,c=exec_cost)  (only if task fits individually on node)
    Nj -> K (cap=task_limit,c=0)
    """
    # map names to contiguous integer ids
    name2id: Dict[str, int] = {}
    def nid(name: str) -> int:
        if name not in name2id:
            name2id[name] = len(name2id)
        return name2id[name]

    S = nid("S")
    K = nid("K")
    for t in tasks:
        nid(f"T:{t['id']}")
    for n in nodes:
        nid(f"N:{n['id']}")

    mcmf = MinCostMaxFlow(len(name2id))

    # S -> Ti
    for t in tasks:
        mcmf.add_edge(S, name2id[f"T:{t['id']}"], 1, 0)

    # Ti -> Nj
    unassignable = set()  # tasks that have zero eligible node edges
    for t in tasks:
        ti = name2id[f"T:{t['id']}"]
        has_edge = False
        for n in nodes:
            cost = exec_cost.get(t["id"], {}).get(n["id"])
            fits = (t["cpu"] <= n["cpu_capacity"] and t["ram"] <= n["ram_capacity"])
            if cost is not None and fits:
                mcmf.add_edge(ti, name2id[f"N:{n['id']}"], 1, int(cost))
                has_edge = True
        if not has_edge:
            unassignable.add(t["id"])

    # Nj -> K
    for n in nodes:
        cap = n.get("task_limit")
        if cap is None:
            cap = fallback_task_limit if fallback_task_limit is not None else _estimate_task_limit(tasks, n)
        mcmf.add_edge(name2id[f"N:{n['id']}"], K, int(cap), 0)

    return mcmf, name2id, S, K, sorted(list(unassignable))


def solve_phase1(tasks: List[dict], nodes: List[dict], exec_cost: Dict[str, Dict[str, int]],
                 fallback_task_limit: Optional[int] = None):
    """Run MCMF and extract assignment & total cost."""
    mcmf, name2id, S, K, unassignable = build_phase1_graph(tasks, nodes, exec_cost, fallback_task_limit)

    total_tasks = len(tasks)
    need_flow = total_tasks

    # If some tasks have no eligible nodes, we already know it's infeasible.
    if unassignable:
        return {
            "valid": False,
            "reason": "Some tasks have no eligible node (resource/cost edge missing).",
            "unassignable_tasks": unassignable,
            "assignments": {},
            "total_cost": None
        }

    flow, total_cost = mcmf.min_cost_flow(S, K, need_flow)
    assignments: Dict[str, str] = {}

    if flow != need_flow:
        # figure out which tasks remained unmatched (for debugging)
        matched_tasks = set()
        for name, u in name2id.items():
            if not name.startswith("T:"):
                continue
            for e in mcmf.g[u]:
                # If reverse capacity > 0 on T->N edge => one unit of flow used
                if e.rev.cap > 0:
                    vname = next((k for k, val in name2id.items() if val == e.v), None)
                    if vname and vname.startswith("N:"):
                        matched_tasks.add(name[2:])
        unmatched = [t["id"] for t in tasks if t["id"] not in matched_tasks]

        return {
            "valid": False,
            "reason": "Not enough capacity on nodes to host all tasks.",
            "unassignable_tasks": unmatched,
            "assignments": assignments,
            "total_cost": None
        }

    # Extract T->N with flow == 1
    id2name = {v: k for k, v in name2id.items()}
    for t in tasks:
        ut = name2id[f"T:{t['id']}"]
        for e in mcmf.g[ut]:
            if e.v == S:
                continue
            if e.rev.cap > 0:  # one unit flowed here
                vname = id2name[e.v]
                if vname.startswith("N:"):
                    assignments[t["id"]] = vname[2:]

    return {
        "valid": True,
        "assignments": assignments,
        "total_cost": int(total_cost)
    }


# ----------------------------------- CLI -------------------------------------

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _dump_json(obj: dict, path: Optional[str]):
    data = json.dumps(obj, ensure_ascii=False, indent=2)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(data + "\n")
    else:
        print(data)

def main():
    p = argparse.ArgumentParser(description="Phase 1 MCMF (Successive Shortest Path) without external libs")
    p.add_argument("--in", dest="inp", help="Input JSON file", required=False)
    p.add_argument("--out", dest="out", help="Output JSON file (optional)")
    p.add_argument("--fallback-task-limit", type=int, default=None,
                   help="Default max tasks per node if node.task_limit is missing")
    p.add_argument("--demo", action="store_true", help="Run with a small built-in demo")
    args = p.parse_args()

    if args.demo:
        tasks = [
            {"id":"T1","cpu":2,"ram":4,"deadline":2},
            {"id":"T2","cpu":1,"ram":2,"deadline":3},
            {"id":"T3","cpu":2,"ram":2,"deadline":3}
        ]
        nodes = [
            {"id":"N1","cpu_capacity":5,"ram_capacity":6,"task_limit":2},
            {"id":"N2","cpu_capacity":3,"ram_capacity":3,"task_limit":2}
        ]
        exec_cost = {
            "T1":{"N1":4,"N2":6},
            "T2":{"N1":3,"N2":2},
            "T3":{"N1":2,"N2":5}
        }
        res = solve_phase1(tasks, nodes, exec_cost, args.fallback_task_limit)
        _dump_json(res, args.out)
        return

    if not args.inp:
        print("Either provide --in <file.json> or use --demo.", file=sys.stderr)
        sys.exit(2)

    data = _load_json(args.inp)
    tasks = data.get("tasks", [])
    nodes = data.get("nodes", [])
    exec_cost = data.get("exec_cost", {})

    # basic validation
    if not tasks or not nodes:
        _dump_json({
            "valid": False,
            "reason": "Missing tasks or nodes in input.",
            "assignments": {},
            "total_cost": None
        }, args.out)
        return

    res = solve_phase1(tasks, nodes, exec_cost, args.fallback_task_limit)
    _dump_json(res, args.out)

if __name__ == "__main__":
    main()
