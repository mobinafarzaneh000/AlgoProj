#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 — SSP with Dynamic task_limit per step (no external libs)

At each step (assign exactly one task):
  task_limit_j = min( floor(cpu_rem_j / min_cpu_task_rem),
                      floor(ram_rem_j / min_ram_task_rem) )
is recomputed and used as capacity of edge N_j -> K for that step.

We rebuild the residual graph every step, send flow=1, extract chosen (task,node),
update residual capacities cpu_rem/ram_rem, and continue.

Input JSON:
{
  "tasks":[{"id":"T1","cpu":2,"ram":4,"deadline":3}, ...],
  "nodes":[{"id":"N1","cpu_capacity":5,"ram_capacity":6}, ...],
  "exec_cost":{"T1":{"N1":3,"N2":5,...}, ...}
}

Output JSON:
{
  "valid": true/false,
  "assignments": {"T1":"N1", ...},
  "total_cost": 17,
  "steps": [ { "chosen_task":"T1","node":"N1","cost":3,
               "cpu_rem":{"N1":3,...},"ram_rem":{"N1":2,...} }, ... ],
  "reason": "... (if invalid)"
}
"""

import json
import argparse
import sys
import heapq
from typing import Dict, List, Tuple, Optional

INF = 10**18

# ------------------------ Min-Cost Max-Flow (SSP) ------------------------

class MinCostMaxFlow:
    class Edge:
        __slots__ = ("u","v","cap","cost","rev")
        def __init__(self, u: int, v: int, cap: int, cost: int, rev: Optional["MinCostMaxFlow.Edge"]):
            self.u = u; self.v = v; self.cap = cap; self.cost = cost; self.rev = rev

    def __init__(self, n: int):
        self.n = n
        self.g: List[List[MinCostMaxFlow.Edge]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: int, cost: int):
        a = MinCostMaxFlow.Edge(u, v, cap, cost, None)
        b = MinCostMaxFlow.Edge(v, u, 0,   -cost, None)
        a.rev = b; b.rev = a
        self.g[u].append(a); self.g[v].append(b)

    def min_cost_flow(self, s: int, t: int, want_flow: int) -> Tuple[int, int]:
        n = self.n
        flow = 0
        cost = 0
        pot = [0]*n  # Johnson potentials; initial costs assumed non-negative

        while flow < want_flow:
            dist = [INF]*n
            parent: List[Optional[MinCostMaxFlow.Edge]] = [None]*n
            dist[s] = 0
            pq = [(0, s)]
            while pq:
                d,u = heapq.heappop(pq)
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
                break

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

# ------------------------ SSP with dynamic task_limit ------------------------

def solve_phase1_dynamic_limit(data: dict, verbose: bool=False):
    tasks = data.get("tasks", [])
    nodes = data.get("nodes", [])
    exec_cost = data.get("exec_cost", {})

    if not tasks or not nodes or not exec_cost:
        return {"valid": False, "assignments": {}, "total_cost": None,
                "reason": "Missing tasks/nodes/exec_cost in input."}

    task_ids = [t["id"] for t in tasks]
    node_ids = [n["id"] for n in nodes]

    cpu_req = {t["id"]: int(t["cpu"]) for t in tasks}
    ram_req = {t["id"]: int(t["ram"]) for t in tasks}
    cpu_rem = {n["id"]: int(n["cpu_capacity"]) for n in nodes}
    ram_rem = {n["id"]: int(n["ram_capacity"]) for n in nodes}

    # سریع: تسکی که «به‌صورت فردی» روی هیچ نودی جا نمی‌شود، از همین ابتدا infeasible است
    for ti in task_ids:
        individually_ok = any(
            (ti in exec_cost and nj in exec_cost[ti]) and
            (cpu_req[ti] <= cpu_rem[nj] and ram_req[ti] <= ram_rem[nj])
            for nj in node_ids
        )
        if not individually_ok:
            return {"valid": False, "assignments": {}, "total_cost": None,
                    "reason": f"Task {ti} has no individually feasible node (under current capacities)."}

    remaining = set(task_ids)
    assignments: Dict[str, str] = {}
    total_cost = 0
    steps_log = []

    def min_reqs():
        if not remaining:
            return None, None
        mc = min(cpu_req[t] for t in remaining)
        mr = min(ram_req[t] for t in remaining)
        return mc, mr

    step = 0
    while remaining:
        step += 1
        mc, mr = min_reqs()
        if mc is None:
            break

        # ساخت گراف گام جاری
        name2id: Dict[str,int] = {}
        def nid(name: str) -> int:
            if name not in name2id:
                name2id[name] = len(name2id)
            return name2id[name]

        S = nid("S"); K = nid("K")
        for t in sorted(remaining):
            nid(f"T:{t}")
        for j in node_ids:
            nid(f"N:{j}")

        mcmf = MinCostMaxFlow(len(name2id))

        # S -> T (cap=1)
        for t in remaining:
            mcmf.add_edge(S, name2id[f"T:{t}"], 1, 0)

        # T -> N (cap=1) فقط اگر از نظر باقیمانده جا شود
        any_edge = False
        for t in remaining:
            for j in node_ids:
                if t in exec_cost and j in exec_cost[t]:
                    if cpu_req[t] <= cpu_rem[j] and ram_req[t] <= ram_rem[j]:
                        mcmf.add_edge(name2id[f"T:{t}"], name2id[f"N:{j}"], 1, int(exec_cost[t][j]))
                        any_edge = True
        if not any_edge:
            return {"valid": False, "assignments": assignments, "total_cost": None,
                    "reason": "No feasible T->N edges exist for remaining tasks (under current residual capacities)."}

        # N -> K با task_limit_j داینامیک
        for j in node_ids:
            if mc <= 0 or mr <= 0:
                Lj = 0
            else:
                Lj = min(cpu_rem[j] // mc, ram_rem[j] // mr)
            if Lj < 0:
                Lj = 0
            mcmf.add_edge(name2id[f"N:{j}"], K, Lj, 0)

        # ارسال 1 واحد جریان
        flow, cost = mcmf.min_cost_flow(S, K, 1)
        if flow != 1:
            return {"valid": False, "assignments": assignments, "total_cost": None,
                    "reason": f"Cannot assign more tasks at step {step}; dynamic limits block further assignment."}

        # استخراج تسک و نود انتخاب‌شده
        id2name = {v:k for k,v in name2id.items()}
        chosen_t = None
        chosen_n = None
        for t in remaining:
            u = name2id[f"T:{t}"]
            for e in mcmf.g[u]:
                # اگر روی T->N جریان رفت، ظرفیت معکوس > 0 می‌شود
                if e.rev.cap > 0:
                    vname = id2name[e.v]
                    if vname.startswith("N:"):
                        chosen_t = t
                        chosen_n = vname[2:]
                        break
            if chosen_t:
                break

        if chosen_t is None or chosen_n is None:
            return {"valid": False, "assignments": assignments, "total_cost": None,
                    "reason": f"Internal error extracting assignment at step {step}."}

        # به‌روزرسانی باقیمانده‌ها
        assignments[chosen_t] = chosen_n
        total_cost += int(exec_cost[chosen_t][chosen_n])
        cpu_rem[chosen_n] -= cpu_req[chosen_t]
        ram_rem[chosen_n] -= ram_req[chosen_t]
        remaining.remove(chosen_t)

        # لاگ این مرحله
        steps_log.append({
            "step": step,
            "chosen_task": chosen_t,
            "node": chosen_n,
            "cost": int(exec_cost[chosen_t][chosen_n]),
            "cpu_rem": {j:int(cpu_rem[j]) for j in node_ids},
            "ram_rem": {j:int(ram_rem[j]) for j in node_ids}
        })

        if verbose:
            print(f"[step {step}] assign {chosen_t} -> {chosen_n} (cost {exec_cost[chosen_t][chosen_n]})")
            print(" cpu_rem:", {j: cpu_rem[j] for j in node_ids})
            print(" ram_rem:", {j: ram_rem[j] for j in node_ids})

    return {"valid": True, "assignments": assignments, "total_cost": int(total_cost), "steps": steps_log}


# ----------------------------------- CLI -------------------------------------

def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _dump_json(obj: dict, path: Optional[str]):
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(s + "\n")
    else:
        print(s)

def main():
    ap = argparse.ArgumentParser(description="Phase 1 — SSP with dynamic task_limit per step (no external libs)")
    ap.add_argument("--in", dest="inp", help="Input JSON file")
    ap.add_argument("--out", dest="out", help="Output JSON file (optional)")
    ap.add_argument("--demo", action="store_true", help="Run built-in demo")
    ap.add_argument("--verbose", action="store_true", help="Print step-by-step log")
    args = ap.parse_args()

    if args.demo:
        data = {
            "tasks": [
                {"id": "T1", "cpu": 2, "ram": 4, "deadline": 3},
                {"id": "T2", "cpu": 1, "ram": 2, "deadline": 4},
                {"id": "T3", "cpu": 3, "ram": 5, "deadline": 5},
                {"id": "T4", "cpu": 2, "ram": 3, "deadline": 6}
            ],
            "nodes": [
                {"id": "N1", "cpu_capacity": 5, "ram_capacity": 6},
                {"id": "N2", "cpu_capacity": 3, "ram_capacity": 4},
                {"id": "N3", "cpu_capacity": 4, "ram_capacity": 6},
                {"id": "N4", "cpu_capacity": 4, "ram_capacity": 5}
            ],
            "exec_cost": {
                "T1": {"N1": 3, "N2": 5, "N3": 6, "N4": 7},
                "T2": {"N1": 2, "N2": 4, "N3": 5, "N4": 6},
                "T3": {"N1": 4, "N2": 6, "N3": 7, "N4": 8},
                "T4": {"N1": 5, "N2": 7, "N3": 6, "N4": 5}
            }
        }
    else:
        if not args.inp:
            print("Provide --in <file.json> or use --demo.", file=sys.stderr)
            sys.exit(2)
        data = _load_json(args.inp)

    res = solve_phase1_dynamic_limit(data, verbose=args.verbose)
    _dump_json(res, args.out)

if __name__ == "__main__":
    main()
