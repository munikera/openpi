#!/usr/bin/env python3
"""
analyze_trace.py  —  Parse a pt.trace.json and produce a real op+kernel breakdown.

Usage:
    python scripts/analyze_trace.py profiler_output/baseline_xpu/*.pt.trace.json
    python scripts/analyze_trace.py profiler_output/baseline_xpu/*.pt.trace.json \
                                    profiler_output/full_no_cast_xpu/*.pt.trace.json

Output per trace:
  • Per-stage GPU device time (from actual kernel durations)
  • Per CPU-op GPU device time (kernel durations attributed via correlation)
  • Per raw kernel name GPU device time
  • Comparison table when two traces are provided

How attribution works:
    CPU op  ──(External id)──▶  xpu_runtime  ──(correlation)──▶  kernel
    (host ts, cpu_op cat)        (urEnqueueKernelLaunch)            (device ts, kernel cat)

    CPU op timestamps  ∈  host clock domain  → used for stage window matching
    Kernel dur         ∈  device clock domain → actual GPU execution time
    The correlation id is the bridge.

Notes:
  • Kernel timestamps are in the device clock domain — NOT comparable to CPU timestamps.
    Stage assignment uses the CPU dispatch timestamp of the launching op instead.
  • Multiple profiler iterations in one trace are all summed and then divided by iter count.
    Iter boundaries are detected from top-level 'sample_actions' user_annotation events.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ─── Load & index ────────────────────────────────────────────────────────────

def load_trace(path: str):
    p = Path(path)
    with open(p) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    return events


def build_index(events):
    """Return dicts needed for attribution."""
    # 1. CPU op External id → (name, ts, dur)
    ext_to_op = {}
    for e in events:
        if e.get("cat") == "cpu_op" and e.get("ph") == "X":
            eid = e["args"].get("External id") or e["args"].get("Ev Idx")
            if eid is not None:
                ext_to_op[eid] = {"name": e["name"], "ts": e["ts"], "dur": e.get("dur", 0)}

    # 2. xpu_runtime or cuda_runtime correlation → External id
    rt_corr_to_ext = {}
    for e in events:
        if e.get("cat") in ("xpu_runtime", "cuda_runtime") and e.get("ph") == "X":
            corr = e["args"].get("correlation")
            ext  = e["args"].get("External id")
            if corr is not None and ext is not None:
                rt_corr_to_ext[corr] = ext

    # 3. Kernel events (actual device execution) — exclude gpu_user_annotation (span annotations)
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]

    # 4. User annotation stage windows (host clock, same domain as cpu_op ts)
    stage_windows = []
    for e in events:
        if e.get("cat") == "user_annotation" and e.get("ph") == "X":
            stage_windows.append({"name": e["name"], "ts": e["ts"], "end": e["ts"] + e.get("dur", 0)})
    stage_windows.sort(key=lambda x: x["ts"])

    return ext_to_op, rt_corr_to_ext, kernels, stage_windows


def count_iterations(stage_windows):
    """Count top-level 'sample_actions' annotations = profiler iterations."""
    n = sum(1 for s in stage_windows if s["name"] == "sample_actions")
    return max(n, 1)


# ─── Attribution ─────────────────────────────────────────────────────────────

def attribute_kernels(ext_to_op, rt_corr_to_ext, kernels, stage_windows):
    """
    For each kernel, find:
      - which CPU op launched it (via correlation)
      - which stage the CPU op was dispatched in
    Return list of attributed kernel records.
    """
    # Build a fast lookup: for a given host timestamp, which stages contain it?
    # Stages can be nested; we want the innermost one.
    # Sort by window size ascending so innermost wins.
    sw_sorted = sorted(stage_windows, key=lambda x: x["end"] - x["ts"])

    records = []
    unmatched = 0

    for k in kernels:
        corr = k["args"].get("correlation")
        ext  = rt_corr_to_ext.get(corr)
        op_info = ext_to_op.get(ext)

        if op_info is None:
            unmatched += 1
            op_name = "unknown"
            cpu_ts  = None
        else:
            op_name = op_info["name"]
            cpu_ts  = op_info["ts"]

        # Find innermost stage containing this CPU dispatch ts
        stage = "other"
        if cpu_ts is not None:
            for sw in sw_sorted:
                if sw["ts"] <= cpu_ts <= sw["end"]:
                    stage = sw["name"]
                    break

        # Shorten raw kernel name for display
        raw = k["name"]
        # keep triton names intact; shorten native xpu:: functor names
        if "triton_" in raw:
            kname = raw  # already compact
        elif "::" in raw:
            # take last two segments: namespace::FunctorName<...>  → clip template args
            parts = raw.rsplit("::", 1)
            kname = parts[-1].split("<")[0]
        else:
            kname = raw.split("<")[0]

        records.append({
            "op":     op_name,
            "stage":  stage,
            "kname":  kname,
            "kname_full": raw,
            "dur_us": k.get("dur", 0),
        })

    if unmatched:
        print(f"  [warn] {unmatched}/{len(kernels)} kernels had no CPU op match", file=sys.stderr)

    return records


# ─── Aggregation ─────────────────────────────────────────────────────────────

def aggregate(records, key, n_iters):
    agg = defaultdict(float)
    cnt = defaultdict(int)
    for r in records:
        agg[r[key]] += r["dur_us"]
        cnt[r[key]] += 1
    result = []
    for k in sorted(agg, key=lambda x: -agg[x]):
        result.append({
            "name":       k,
            "total_us":   agg[k],
            "per_iter_ms": agg[k] / n_iters / 1000,
            "calls_total": cnt[k],
            "calls_iter":  cnt[k] / n_iters,
        })
    return result


# ─── Printing ─────────────────────────────────────────────────────────────────

def print_table(title, rows, name_col_w=55, top=40):
    print(f"\n{'─'*85}")
    print(f"  {title}")
    print(f"{'─'*85}")
    total = sum(r["per_iter_ms"] for r in rows)
    print(f"  {'Name':{name_col_w}}  {'ms/iter':>9}  {'%tot':>6}  {'calls/iter':>10}")
    print(f"  {'-'*name_col_w}  {'-'*9}  {'-'*6}  {'-'*10}")
    for r in rows[:top]:
        pct = 100 * r["per_iter_ms"] / total if total else 0
        print(f"  {r['name']:{name_col_w}}  {r['per_iter_ms']:>9.3f}  {pct:>5.1f}%  {r['calls_iter']:>10.1f}")
    if len(rows) > top:
        rest = sum(r["per_iter_ms"] for r in rows[top:])
        print(f"  {'... ({} more)'.format(len(rows)-top):{name_col_w}}  {rest:>9.3f}")
    print(f"  {'─'*name_col_w}  {'-'*9}")
    print(f"  {'TOTAL (all kernels)':{name_col_w}}  {total:>9.3f}")


def print_comparison(title, rows_a, rows_b, label_a, label_b, name_col_w=55, top=40):
    print(f"\n{'─'*100}")
    print(f"  {title}")
    print(f"{'─'*100}")
    # Union of names
    map_a = {r["name"]: r for r in rows_a}
    map_b = {r["name"]: r for r in rows_b}
    all_names = sorted(set(map_a) | set(map_b),
                       key=lambda n: -(map_a.get(n, {}).get("per_iter_ms", 0) +
                                       map_b.get(n, {}).get("per_iter_ms", 0)))
    tot_a = sum(r["per_iter_ms"] for r in rows_a)
    tot_b = sum(r["per_iter_ms"] for r in rows_b)
    print(f"  {'Name':{name_col_w}}  {label_a:>10}  {label_b:>10}  {'Δ ms':>8}  {'Δ%':>6}")
    print(f"  {'-'*name_col_w}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}")
    for name in all_names[:top]:
        a = map_a.get(name, {}).get("per_iter_ms", 0)
        b = map_b.get(name, {}).get("per_iter_ms", 0)
        delta = b - a
        pct = 100 * delta / a if a > 0 else float("inf")
        flag = "  ❌" if delta > 0.5 else ("  ✅" if delta < -0.5 else "")
        print(f"  {name:{name_col_w}}  {a:>10.3f}  {b:>10.3f}  {delta:>+8.3f}  {pct:>+5.1f}%{flag}")
    print(f"  {'─'*name_col_w}  {'-'*10}  {'-'*10}  {'-'*8}")
    delta_tot = tot_b - tot_a
    pct_tot = 100 * delta_tot / tot_a if tot_a else 0
    print(f"  {'TOTAL':{name_col_w}}  {tot_a:>10.3f}  {tot_b:>10.3f}  {delta_tot:>+8.3f}  {pct_tot:>+5.1f}%")


# ─── Per-trace analysis ───────────────────────────────────────────────────────

def analyze(path, label=None):
    label = label or Path(path).parent.name
    print(f"\n{'═'*85}")
    print(f"  TRACE: {label}")
    print(f"  File:  {path}")

    events = load_trace(path)
    ext_to_op, rt_corr_to_ext, kernels, stage_windows = build_index(events)
    n_iters = count_iterations(stage_windows)
    print(f"  Profiler iterations detected: {n_iters}")
    print(f"  Total kernel events: {len(kernels)}")
    total_kernel_us = sum(k.get("dur", 0) for k in kernels)
    print(f"  Total GPU device time: {total_kernel_us/n_iters/1000:.2f} ms/iter")

    records = attribute_kernels(ext_to_op, rt_corr_to_ext, kernels, stage_windows)

    # 1. By stage (top-level only: sample_actions, stage1_*, stage2_*, stage3_*)
    top_stages = ["sample_actions", "stage0_preprocess", "stage1_embed_prefix",
                  "stage2_prefix_fwd", "stage3_step0", "stage3a_embed_suffix",
                  "stage3b_expert_fwd", "other"]
    stage_rows = aggregate(records, "stage", n_iters)

    # 2. By CPU op name
    op_rows = aggregate(records, "op", n_iters)

    # 3. By short kernel name
    kname_rows = aggregate(records, "kname", n_iters)

    # ── Stage breakdown  (collapse stage3_stepN → stage3_denoise)
    stage_map = defaultdict(float)
    stage_call_map = defaultdict(float)
    for r in records:
        s = r["stage"]
        if s.startswith("stage3_step"):
            s = "stage3_denoise (×10)"
        elif s.startswith("stage3a_"):
            s = "stage3a_embed_suffix (×10)"
        elif s.startswith("stage3b_"):
            s = "stage3b_expert_fwd (×10)"
        stage_map[s] += r["dur_us"]
        stage_call_map[s] += 1
    stage_rows_collapsed = sorted(
        [{"name": k, "per_iter_ms": v/n_iters/1000, "calls_iter": stage_call_map[k]/n_iters}
         for k, v in stage_map.items()],
        key=lambda x: -x["per_iter_ms"]
    )

    print_table(f"GPU device time by STAGE  (ms/iter, {n_iters} iters)",
                stage_rows_collapsed, name_col_w=40, top=20)
    print_table(f"GPU device time by CPU OP  (ms/iter, {n_iters} iters)",
                op_rows, name_col_w=40, top=30)
    print_table(f"GPU device time by KERNEL NAME  (ms/iter, {n_iters} iters)",
                kname_rows, name_col_w=65, top=40)

    return {
        "label":       label,
        "n_iters":     n_iters,
        "records":     records,
        "stage_rows":  stage_rows_collapsed,
        "op_rows":     op_rows,
        "kname_rows":  kname_rows,
        "total_ms":    total_kernel_us / n_iters / 1000,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="One or two .pt.trace.json files")
    ap.add_argument("--top", type=int, default=40, help="Max rows per table (default 40)")
    args = ap.parse_args()

    results = []
    for path in args.traces[:2]:
        results.append(analyze(path))

    if len(results) == 2:
        a, b = results
        print(f"\n\n{'═'*100}")
        print(f"  COMPARISON:  {a['label']}  →  {b['label']}")
        print(f"  Total GPU: {a['total_ms']:.2f} ms/iter  →  {b['total_ms']:.2f} ms/iter  "
              f"(Δ {b['total_ms']-a['total_ms']:+.2f} ms)")
        print(f"{'═'*100}")

        print_comparison("By STAGE", a["stage_rows"], b["stage_rows"],
                         a["label"][:10], b["label"][:10], name_col_w=40)
        print_comparison("By CPU OP", a["op_rows"], b["op_rows"],
                         a["label"][:10], b["label"][:10], name_col_w=40)
        print_comparison("By KERNEL NAME", a["kname_rows"], b["kname_rows"],
                         a["label"][:10], b["label"][:10], name_col_w=65)


if __name__ == "__main__":
    main()
