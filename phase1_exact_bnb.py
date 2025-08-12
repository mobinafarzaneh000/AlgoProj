
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 — Exact solver (Branch & Bound) for Task-to-Node assignment
- رعایت دقیق قیود: Σcpu ≤ cpu_capacity_j ، Σram ≤ ram_capacity_j ، و task_limit_j (اگر باشد)
- هدف: کمینه‌سازی مجموع exec_cost
- بدون هیچ کتابخانه خارجی

ورودی:
{
  "tasks":[{"id":"T1","cpu":2,"ram":4,"deadline":3}, ...],
  "nodes":[{"id":"N1","cpu_capacity":5,"ram_capacity":6,"task_limit":3}, ...],
  "exec_cost":{"T1":{"N1":4,"N2":2}, ...}
}

خروجی:
{
  "valid": true/false,
  "assignments": {"T1":"N2", ...},
  "total_cost": <int>,
  "reason": "... (اگر ناممکن)"
}
"""

import json, sys, argparse
from typing import Dict, List, Tuple, Optional

INF = 10**15

def _load_json(p:str)->dict:
    with open(p,"r",encoding="utf-8") as f: return json.load(f)

def _dump_json(o:dict, p:Optional[str]):
    s=json.dumps(o, ensure_ascii=False, indent=2)
    if p:
        with open(p,"w",encoding="utf-8") as f: f.write(s+"\n")
    else:
        print(s)

def solve_phase1_exact(data: dict) -> dict:
    tasks = data.get("tasks", [])
    nodes = data.get("nodes", [])
    exec_cost: Dict[str,Dict[str,int]] = data.get("exec_cost", {}) or {}
    if not tasks or not nodes or not exec_cost:
        return {"valid": False, "assignments": {}, "total_cost": None,
                "reason": "Missing tasks/nodes/exec_cost"}

    task_ids = [t["id"] for t in tasks]
    node_ids = [n["id"] for n in nodes]

    cpu_req = {t["id"]: int(t["cpu"]) for t in tasks}
    ram_req = {t["id"]: int(t["ram"]) for t in tasks}
    cpu_cap = {n["id"]: int(n["cpu_capacity"]) for n in nodes}
    ram_cap = {n["id"]: int(n["ram_capacity"]) for n in nodes}
    task_limit = {n["id"]: int(n.get("task_limit", 10**9)) for n in nodes}

    # پیش‌بررسی: هر تسک دست‌کم یک نود ممکن داشته باشد (از نظر فردی)
    for ti in task_ids:
        ok = False
        for nj in node_ids:
            if ti in exec_cost and nj in exec_cost[ti]:
                if cpu_req[ti] <= cpu_cap[nj] and ram_req[ti] <= ram_cap[nj]:
                    ok = True; break
        if not ok:
            return {"valid": False, "assignments": {}, "total_cost": None,
                    "reason": f"Task {ti} has no individually feasible node."}

    # مرتب‌سازی تسک‌ها برای برش بهتر: اول کمترین تعداد گزینه‌ی ممکن، بعد پرهزینه‌ترین
    def feasible_nodes_of(ti:str) -> List[str]:
        out=[]
        for nj in node_ids:
            if ti in exec_cost and nj in exec_cost[ti]:
                # فرداً شدنی روی ظرفیت اولیه
                if cpu_req[ti] <= cpu_cap[nj] and ram_req[ti] <= ram_cap[nj]:
                    out.append(nj)
        return out

    candidates = {ti: feasible_nodes_of(ti) for ti in task_ids}
    order = sorted(task_ids, key=lambda x: (len(candidates[x]), -min(exec_cost[x][j] for j in candidates[x])))

    # Lower bound: مجموع کمترین هزینه‌ی ممکن هر تسک (صرف‌نظر از ظرفیت)
    min_cost_per_task = {ti: min(exec_cost[ti][j] for j in candidates[ti]) for ti in task_ids}
    global_best = {"cost": INF, "assign": {}}

    # وضعیت باقیمانده منابع هر نود
    cpu_rem = {j: cpu_cap[j] for j in node_ids}
    ram_rem = {j: ram_cap[j] for j in node_ids}
    cnt_rem = {j: task_limit[j] for j in node_ids}

    assign: Dict[str,str] = {}

    # مرتب‌سازی گزینه‌های هر تسک به ترتیب کم‌هزینه‌ترین نودها
    sorted_nodes_for = {ti: sorted(candidates[ti], key=lambda j: exec_cost[ti][j]) for ti in task_ids}

    def dfs(idx: int, cur_cost: int):
        nonlocal global_best, assign, cpu_rem, ram_rem, cnt_rem

        # برش بر اساس هزینه
        # حد پایین = cur_cost + جمعِ حداقل‌هزینه‌ی تسک‌های باقی‌مانده
        if cur_cost >= global_best["cost"]:
            return
        lb = cur_cost
        for k in range(idx, len(order)):
            lb += min_cost_per_task[order[k]]
            if lb >= global_best["cost"]:
                return

        # تمام شد
        if idx == len(order):
            # راه‌حل بهتر
            global_best["cost"] = cur_cost
            global_best["assign"] = assign.copy()
            return

        ti = order[idx]
        # سعی کن روی نودهای کم‌هزینه‌تر اول امتحان کنی
        for nj in sorted_nodes_for[ti]:
            # امکان‌پذیری منابع
            if cpu_req[ti] <= cpu_rem[nj] and ram_req[ti] <= ram_rem[nj] and cnt_rem[nj] > 0:
                # انتخاب
                assign[ti] = nj
                cpu_rem[nj] -= cpu_req[ti]
                ram_rem[nj] -= ram_req[ti]
                cnt_rem[nj] -= 1

                dfs(idx + 1, cur_cost + exec_cost[ti][nj])

                # backtrack
                cnt_rem[nj] += 1
                ram_rem[nj] += ram_req[ti]
                cpu_rem[nj] += cpu_req[ti]
                del assign[ti]

        # اگر هیچ گزینه‌ای نشد، برش
        return

    dfs(0, 0)

    if global_best["cost"] >= INF:
        return {"valid": False, "assignments": {}, "total_cost": None,
                "reason": "No feasible assignment under CPU/RAM/task_limit constraints."}

    return {"valid": True, "assignments": global_best["assign"], "total_cost": int(global_best["cost"])}

# ---------------- CLI ----------------
def main():
    ap=argparse.ArgumentParser(description="Phase 1 — Exact Task-to-Node assignment (Branch & Bound)")
    ap.add_argument("--in", dest="inp", help="Input JSON")
    ap.add_argument("--out", dest="out", help="Output JSON (optional)")
    args=ap.parse_args()
    if not args.inp:
        print("Provide --in <file.json>", file=sys.stderr); sys.exit(2)
    data=_load_json(args.inp)
    res=solve_phase1_exact(data)
    _dump_json(res, args.out)

if __name__=="__main__":
    main()
