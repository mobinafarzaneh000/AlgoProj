#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 — SSP (Successive Shortest Path) with:
  • Persistent residual graph (reverse edges persist)
  • Dynamic task_limit_j update before each augmentation
  • T->N edges enabled by individual feasibility on initial node capacities
  • Step-by-step diff logging: step_cost, added, moved, assignments_after

Run:
  python phase1_ssp_dynamic_limit_persistent_final.py --in data.json --verbose
Or demo (negative-edge case):
  python phase1_ssp_dynamic_limit_persistent_final.py --demo_neg --verbose
"""

import json, sys, argparse, heapq
from typing import Dict, List, Tuple, Optional

INF = 10**18

# ---------------- Min-Cost Max-Flow (SSP + Potentials) ----------------
class MinCostMaxFlow:
    class Edge:
        __slots__ = ("u","v","cap","cost","rev")
        def __init__(self, u:int, v:int, cap:int, cost:int, rev:Optional["MinCostMaxFlow.Edge"]):
            self.u=u; self.v=v; self.cap=cap; self.cost=cost; self.rev=rev

    def __init__(self, n:int):
        self.n=n
        self.g: List[List[MinCostMaxFlow.Edge]]=[[] for _ in range(n)]

    def add_edge(self,u:int,v:int,cap:int,cost:int)->"MinCostMaxFlow.Edge":
        a=MinCostMaxFlow.Edge(u,v,cap,cost,None)
        b=MinCostMaxFlow.Edge(v,u,0,-cost,None)
        a.rev=b; b.rev=a
        self.g[u].append(a); self.g[v].append(b)
        return a

    def min_cost_flow(self,s:int,t:int,want:int)->Tuple[int,int]:
        n=self.n; flow=0; cost=0; pot=[0]*n
        while flow<want:
            dist=[INF]*n
            parent: List[Optional[MinCostMaxFlow.Edge]]=[None]*n
            dist[s]=0
            pq=[(0,s)]
            while pq:
                d,u=heapq.heappop(pq)
                if d!=dist[u]: continue
                for e in self.g[u]:
                    if e.cap<=0: continue
                    nd=d+e.cost+pot[u]-pot[e.v]
                    if nd<dist[e.v]:
                        dist[e.v]=nd; parent[e.v]=e
                        heapq.heappush(pq,(nd,e.v))
            if parent[t] is None: break
            for v in range(n):
                if dist[v]<INF: pot[v]+=dist[v]
            add=want-flow; v=t
            while v!=s:
                e=parent[v]; add=min(add,e.cap); v=e.u
            v=t
            while v!=s:
                e=parent[v]; e.cap-=add; e.rev.cap+=add; cost+=add*e.cost; v=e.u
            flow+=add
        return flow,cost

# ---------------------- Solver (persistent + dynamic limits) -------------------
def solve_phase1_ssp_persistent_final(data: dict, verbose: bool=False):
    tasks=data.get("tasks",[])
    nodes=data.get("nodes",[])
    exec_cost=data.get("exec_cost",{})
    if not tasks or not nodes or not exec_cost:
        return {"valid":False,"assignments":{},"total_cost":None,"reason":"Missing tasks/nodes/exec_cost"}

    task_ids=[t["id"] for t in tasks]
    node_ids=[n["id"] for n in nodes]
    cpu_req={t["id"]:int(t["cpu"]) for t in tasks}
    ram_req={t["id"]:int(t["ram"]) for t in tasks}
    cpu_cap={n["id"]:int(n["cpu_capacity"]) for n in nodes}
    ram_cap={n["id"]:int(n["ram_capacity"]) for n in nodes}

    # quick individual feasibility check
    for ti in task_ids:
        ok = any(
            (ti in exec_cost and nj in exec_cost[ti]) and
            (cpu_req[ti] <= cpu_cap[nj] and ram_req[ti] <= ram_cap[nj])
            for nj in node_ids
        )
        if not ok:
            return {"valid":False,"assignments":{},"total_cost":None,
                    "reason":f"Task {ti} has no individually feasible node."}

    # Build ids
    name2id: Dict[str,int] = {}
    def nid(name:str)->int:
        if name not in name2id: name2id[name]=len(name2id)
        return name2id[name]

    S=nid("S"); K=nid("K")
    for t in task_ids: nid(f"T:{t}")
    for j in node_ids: nid(f"N:{j}")

    mcmf=MinCostMaxFlow(len(name2id))

    # S -> T
    for t in task_ids:
        mcmf.add_edge(S, name2id[f"T:{t}"], 1, 0)

    # T -> N (store handles), enabled iff individually feasible on INITIAL capacities
    edge_TN: Dict[Tuple[str,str], MinCostMaxFlow.Edge]={}
    for t in task_ids:
        for j in node_ids:
            if t in exec_cost and j in exec_cost[t] and cpu_req[t] <= cpu_cap[j] and ram_req[t] <= ram_cap[j]:
                e = mcmf.add_edge(name2id[f"T:{t}"], name2id[f"N:{j}"], 1, int(exec_cost[t][j]))
                edge_TN[(t,j)] = e

    # N -> K (store handles; cap updated per step)
    edge_NK: Dict[str, MinCostMaxFlow.Edge]={}
    for j in node_ids:
        e = mcmf.add_edge(name2id[f"N:{j}"], K, 0, 0)
        edge_NK[j]=e

    # helpers
    def current_assignments()->Dict[str,str]:
        id2name={v:k for k,v in name2id.items()}
        assign={}
        for t in task_ids:
            u=name2id[f"T:{t}"]
            for e in mcmf.g[u]:
                if e.cost>=0 and e.rev.cap>0:
                    vname=id2name[e.v]
                    if vname.startswith("N:"):
                        assign[t]=vname[2:]
                        break
        return assign

    def recompute_residual_resources(assign: Dict[str,str]):
        cpu_rem={j:cpu_cap[j] for j in node_ids}
        ram_rem={j:ram_cap[j] for j in node_ids}
        for t,j in assign.items():
            cpu_rem[j]-=cpu_req[t]; ram_rem[j]-=ram_req[t]
        return cpu_rem, ram_rem

    total_cost=0
    steps=[]
    assigned=current_assignments()
    total_flow=len(assigned)

    while total_flow < len(task_ids):
        assigned=current_assignments()
        total_flow=len(assigned)
        remaining=[t for t in task_ids if t not in assigned]
        if not remaining:
            break

        mc=min(cpu_req[t] for t in remaining)
        mr=min(ram_req[t] for t in remaining)
        cpu_rem, ram_rem = recompute_residual_resources(assigned)

        # --- (1) Enable/disable T->N forward caps ---
        # Keep any *unused* T->N that is individually feasible on INITIAL caps enabled (cap=1).
        for (t,j), e in edge_TN.items():
            if e.rev.cap > 0:
                e.cap = 0  # this T->N is already used in the flow
            else:
                individually_fits = (cpu_req[t] <= cpu_cap[j] and ram_req[t] <= ram_cap[j])
                e.cap = 1 if individually_fits else 0

        # --- (2) Update N->K caps using dynamic limits ---
        used_slots={j:0 for j in node_ids}
        for t,j in assigned.items():
            used_slots[j]+=1
        for j in node_ids:
            if mc<=0 or mr<=0:
                desired=0
            else:
                desired = min(cpu_rem[j]//mc, ram_rem[j]//mr)
            edge_NK[j].cap = max(0, desired)

        # --- (3) Augment 1 unit on persistent graph (DIFF LOG) ---
        prev_assigned = assigned.copy()
        prev_total = total_cost
        flow, cost = mcmf.min_cost_flow(S, K, 1)
        if flow != 1:
            return {"valid":False,"assignments":assigned,"total_cost":None,
                    "reason":"No augmenting path under dynamic limits."}
        total_cost += cost

        # --- (4) Diff-based logging ---
        new_assigned = current_assignments()
        cpu_rem, ram_rem = recompute_residual_resources(new_assigned)

        added=[]
        moved=[]
        for t in new_assigned:
            if t not in prev_assigned:
                added.append({"task": t, "node": new_assigned[t]})
            elif prev_assigned[t] != new_assigned[t]:
                moved.append({"task": t, "from": prev_assigned[t], "to": new_assigned[t]})

        steps.append({
            "step_cost": int(total_cost - prev_total),
            "added": added,
            "moved": moved,
            "assignments_after": {t: new_assigned[t] for t in sorted(new_assigned)},
            "cpu_rem": {j:int(cpu_rem[j]) for j in node_ids},
            "ram_rem": {j:int(ram_rem[j]) for j in node_ids}
        })

        assigned = new_assigned
        if verbose:
            print(f"[+1] step_cost={int(total_cost - prev_total)}, total_cost={total_cost}, assigned={assigned}")

    assigned=current_assignments()
    if len(assigned)!=len(task_ids):
        return {"valid":False,"assignments":assigned,"total_cost":None,"reason":"Not all tasks assigned."}
    return {"valid":True,"assignments":assigned,"total_cost":int(total_cost),"steps":steps}

# ---------------------------------- CLI ----------------------------------
def _load_json(p:str)->dict:
    with open(p,"r",encoding="utf-8") as f: return json.load(f)
def _dump_json(o:dict, p:Optional[str]):
    s=json.dumps(o,ensure_ascii=False,indent=2)
    if p:
        with open(p,"w",encoding="utf-8") as f: f.write(s+"\n")
    else:
        print(s)

def main():
    ap=argparse.ArgumentParser(description="Phase 1 SSP persistent + dynamic limits (final with diff logging)")
    ap.add_argument("--in",dest="inp",help="Input JSON file")
    ap.add_argument("--out",dest="out",help="Output JSON (optional)")
    ap.add_argument("--verbose",action="store_true")
    ap.add_argument("--demo_neg",action="store_true",help="Run the negative-edge demo case")
    args=ap.parse_args()

    if args.demo_neg:
        data={
          "tasks":[
            {"id":"T1","cpu":1,"ram":1,"deadline":10},
            {"id":"T2","cpu":1,"ram":1,"deadline":10},
            {"id":"T3","cpu":1,"ram":1,"deadline":10}
          ],
          "nodes":[
            {"id":"N1","cpu_capacity":2,"ram_capacity":2},
            {"id":"N2","cpu_capacity":1,"ram_capacity":1}
          ],
          "exec_cost":{
            "T1":{"N1":1,"N2":100},
            "T2":{"N1":2,"N2":3},
            "T3":{"N1":2,"N2":100}
          }
        }
    else:
        if not args.inp:
            print("Provide --in <file.json> or use --demo_neg", file=sys.stderr); sys.exit(2)
        data=_load_json(args.inp)

    res=solve_phase1_ssp_persistent_final(data, verbose=args.verbose)
    _dump_json(res, args.out)

if __name__=="__main__":
    main()
