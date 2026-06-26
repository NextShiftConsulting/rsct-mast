"""
diagnose_mast.py — Run GeoCert diagnosis on all 61 MAST traces.

Produces:
  1. Per-trace summary table (top GeoCert diagnosis + confidence)
  2. Cross-tabulation: MAST category × GeoCert category
  3. results/mast_diagnosis.json — full results

The cross-tabulation is the key research finding: it shows whether MAST's
behavioural taxonomy and GeoCert's evaluation taxonomy discover the same
failure structure (convergent) or different structure (orthogonal).

Usage:
    python diagnose_mast.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from load_mast import load_all_traces, get_active_categories, MAST_CATEGORIES
from diagnose import diagnose_trace

RESULTS_DIR = Path(__file__).parent / "results"

# Short display labels for MAST categories
MAST_SHORT = {
    "FC1_Specification": "FC1",
    "FC2_Misalignment": "FC2",
    "FC3_Verification": "FC3",
}

# All GeoCert categories in display order
GC_CATS = ["GC1", "GC2", "GC3"]
GC_CAT_NAMES = {
    "GC1": "GC1 Label Construction",
    "GC2": "GC2 Decomposition Reduction",
    "GC3": "GC3 Deployment Translation",
}


def run_all_diagnoses(traces: list) -> list:
    """Run diagnose_trace on every loaded trace and return enriched records."""
    results = []
    for trace in traces:
        result = diagnose_trace(
            trajectory=trace["trajectory"],
            source=trace["source"],
        )
        top = result["diagnosis"]["candidates"][0]
        results.append({
            "instance_id": trace["instance_id"],
            "source": trace["source"],
            "n_steps": trace["n_steps"],
            "mast_active_categories": sorted(get_active_categories(trace["annotations"])),
            "mast_annotation_count": sum(trace["annotations"].values()),
            "top_gc_mode": top["mode"],
            "top_gc_name": top["name"],
            "top_gc_category": top["category"],
            "top_gc_confidence": top["confidence"],
            "top3_candidates": result["diagnosis"]["candidates"],
            "geocert_signals": result["geocert_signals"],
            "trace_signals": result["signals"],
            "summary": result["summary"],
        })
    return results


def print_summary_table(diagnosed: list) -> None:
    """Print a per-trace summary table to stdout."""
    print("\nPER-TRACE SUMMARY TABLE")
    print("=" * 90)
    hdr = f"  {'ID':>30}  {'Src':>9}  {'Steps':>5}  {'MAST cats':>15}  {'Top GC':>8}  {'Conf':>5}"
    print(hdr)
    print("  " + "-" * 86)
    for r in diagnosed:
        mast_cats = ",".join(MAST_SHORT.get(c, c) for c in r["mast_active_categories"]) or "none"
        iid = r["instance_id"][-30:] if len(r["instance_id"]) > 30 else r["instance_id"]
        print(
            f"  {iid:>30}  {r['source']:>9}  {r['n_steps']:>5}  "
            f"{mast_cats:>15}  {r['top_gc_mode']:>8}  {r['top_gc_confidence']:>5.3f}"
        )


def build_cross_tabulation(diagnosed: list) -> dict:
    """
    Build a cross-tabulation of MAST category vs GeoCert category.

    Returns:
        {
            "counts": {mast_cat: {gc_cat: int}},   # raw counts
            "row_pct": {mast_cat: {gc_cat: float}}, # % within MAST row
            "col_pct": {gc_cat: {mast_cat: float}}, # % within GC column
            "marginals": {"mast": {cat: int}, "gc": {cat: int}},
        }
    """
    mast_cats = list(MAST_CATEGORIES.keys()) + ["baseline"]

    # Raw counts: for traces with no MAST failure, treat as "baseline"
    counts: dict = {mc: defaultdict(int) for mc in mast_cats}
    gc_marginals: dict = defaultdict(int)

    for r in diagnosed:
        gc_cat = r["top_gc_category"]  # e.g. "GC1", "GC2", "GC3"
        gc_marginals[gc_cat] += 1

        active_mast = r["mast_active_categories"]
        if not active_mast:
            counts["baseline"][gc_cat] += 1
        else:
            for mc in active_mast:
                counts[mc][gc_cat] += 1

    # Row percentages (within each MAST category)
    row_pct: dict = {}
    for mc, gc_dist in counts.items():
        total = sum(gc_dist.values())
        if total == 0:
            row_pct[mc] = {gc: 0.0 for gc in GC_CATS}
        else:
            row_pct[mc] = {gc: round(gc_dist.get(gc, 0) / total * 100, 1) for gc in GC_CATS}

    # Column percentages (within each GC category)
    col_pct: dict = {gc: {} for gc in GC_CATS}
    for gc in GC_CATS:
        total_gc = gc_marginals.get(gc, 0)
        for mc in mast_cats:
            n = counts[mc].get(gc, 0)
            col_pct[gc][mc] = round(n / total_gc * 100, 1) if total_gc else 0.0

    mast_marginals = {mc: sum(counts[mc].values()) for mc in mast_cats}

    return {
        "counts": {mc: dict(gc_dist) for mc, gc_dist in counts.items()},
        "row_pct": row_pct,
        "col_pct": col_pct,
        "marginals": {
            "mast": mast_marginals,
            "gc": dict(gc_marginals),
        },
    }


def print_cross_tabulation(xtab: dict) -> None:
    """Print the cross-tabulation tables to stdout."""
    counts = xtab["counts"]
    row_pct = xtab["row_pct"]
    col_pct = xtab["col_pct"]
    mast_marginals = xtab["marginals"]["mast"]
    gc_marginals = xtab["marginals"]["gc"]

    mast_rows = ["FC1_Specification", "FC2_Misalignment", "FC3_Verification", "baseline"]

    print("\n" + "=" * 70)
    print("CROSS-TABULATION: MAST Category x GeoCert Category")
    print("=" * 70)

    # --- Raw counts ---
    print("\nRAW COUNTS  (each MAST-annotated trace may appear in multiple MAST rows)")
    print(f"  {'MAST category':<22}  {'GC1':>6}  {'GC2':>6}  {'GC3':>6}  {'Total':>7}")
    print("  " + "-" * 55)
    for mc in mast_rows:
        row = counts.get(mc, {})
        total = mast_marginals.get(mc, 0)
        gc1 = row.get("GC1", 0)
        gc2 = row.get("GC2", 0)
        gc3 = row.get("GC3", 0)
        label = mc.replace("_", " ")[:22]
        print(f"  {label:<22}  {gc1:>6}  {gc2:>6}  {gc3:>6}  {total:>7}")
    print("  " + "-" * 55)
    print(
        f"  {'GC column total':<22}  "
        f"{gc_marginals.get('GC1', 0):>6}  "
        f"{gc_marginals.get('GC2', 0):>6}  "
        f"{gc_marginals.get('GC3', 0):>6}"
    )

    # --- Row percentages ---
    print("\nROW % (what fraction of each MAST-category's traces map to each GC category)")
    print(f"  {'MAST category':<22}  {'GC1 %':>7}  {'GC2 %':>7}  {'GC3 %':>7}")
    print("  " + "-" * 50)
    for mc in mast_rows:
        rp = row_pct.get(mc, {})
        label = mc.replace("_", " ")[:22]
        print(
            f"  {label:<22}  {rp.get('GC1', 0.0):>7.1f}  "
            f"{rp.get('GC2', 0.0):>7.1f}  {rp.get('GC3', 0.0):>7.1f}"
        )

    # --- Column percentages ---
    print("\nCOL % (what fraction of each GC-category's traces come from each MAST category)")
    print(f"  {'GC category':<24}  {'FC1 %':>7}  {'FC2 %':>7}  {'FC3 %':>7}  {'base %':>7}")
    print("  " + "-" * 58)
    for gc in GC_CATS:
        cp = col_pct.get(gc, {})
        label = GC_CAT_NAMES.get(gc, gc)[:24]
        print(
            f"  {label:<24}  "
            f"{cp.get('FC1_Specification', 0.0):>7.1f}  "
            f"{cp.get('FC2_Misalignment', 0.0):>7.1f}  "
            f"{cp.get('FC3_Verification', 0.0):>7.1f}  "
            f"{cp.get('baseline', 0.0):>7.1f}"
        )


def interpret_cross_tabulation(xtab: dict, diagnosed: list) -> str:
    """
    Generate a short interpretive paragraph for the cross-tabulation.

    Checks whether the MAST → GeoCert mapping is convergent (same structure)
    or orthogonal (different structure).
    """
    row_pct = xtab["row_pct"]
    counts = xtab["counts"]

    lines = []

    # Find dominant GC category for each MAST category
    dominant: dict = {}
    for mc in ["FC1_Specification", "FC2_Misalignment", "FC3_Verification"]:
        rp = row_pct.get(mc, {})
        if rp:
            dom_gc = max(rp, key=lambda g: rp[g])
            dominant[mc] = (dom_gc, rp[dom_gc])

    lines.append("INTERPRETATION")
    lines.append("-" * 50)
    for mc, (dom_gc, pct) in dominant.items():
        label = mc.replace("FC1_Specification", "FC1 (Specification)") \
                  .replace("FC2_Misalignment", "FC2 (Misalignment)") \
                  .replace("FC3_Verification", "FC3 (Verification)")
        lines.append(
            f"  {label} traces predominantly map to {dom_gc} "
            f"({pct:.0f}% of row)."
        )

    # Check convergence: are the three MAST categories each dominated by a
    # different GC category?
    dom_gc_set = {dom_gc for (dom_gc, _) in dominant.values()}
    if len(dom_gc_set) == 3:
        lines.append(
            "\n  CONVERGENT STRUCTURE: Each MAST category maps to a distinct "
            "GeoCert category, suggesting the two taxonomies are indexing the "
            "same underlying failure space."
        )
    elif len(dom_gc_set) == 1:
        lines.append(
            f"\n  ORTHOGONAL STRUCTURE: All three MAST categories map to the "
            f"same GeoCert category ({list(dom_gc_set)[0]}). The two taxonomies "
            f"are capturing different axes of failure."
        )
    else:
        lines.append(
            "\n  PARTIAL OVERLAP: MAST categories partially but not fully map "
            "to distinct GeoCert categories — the taxonomies share some but not "
            "all structure."
        )

    # Confidence distribution
    confs = [r["top_gc_confidence"] for r in diagnosed]
    lines.append(
        f"\n  Diagnosis confidence — mean: {np.mean(confs):.3f}  "
        f"min: {np.min(confs):.3f}  max: {np.max(confs):.3f}"
    )
    lines.append(
        "  (Low confidence reflects that MAST traces are execution-behaviour "
        "data mapped through a heuristic bridge to GeoCert's evaluation-failure "
        "signal space.  Treat diagnosis as indicative, not definitive.)"
    )

    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading MAST traces...")
    traces = load_all_traces()
    print(f"  {len(traces)} traces loaded "
          f"({sum(1 for t in traces if t['source'] == 'AG2')} AG2, "
          f"{sum(1 for t in traces if t['source'] == 'HyperAgent')} HyperAgent)")

    print("\nRunning GeoCert diagnosis on all traces...")
    diagnosed = run_all_diagnoses(traces)
    print(f"  Diagnosed {len(diagnosed)} traces.")

    # -----------------------------------------------------------------------
    # 1. Per-trace summary table
    # -----------------------------------------------------------------------
    print_summary_table(diagnosed)

    # -----------------------------------------------------------------------
    # 2. Cross-tabulation
    # -----------------------------------------------------------------------
    xtab = build_cross_tabulation(diagnosed)
    print_cross_tabulation(xtab)

    # -----------------------------------------------------------------------
    # 3. Interpretation
    # -----------------------------------------------------------------------
    interpretation = interpret_cross_tabulation(xtab, diagnosed)
    print("\n" + "=" * 70)
    print(interpretation)

    # -----------------------------------------------------------------------
    # 4. GC mode frequency table
    # -----------------------------------------------------------------------
    from collections import Counter
    mode_freq = Counter(r["top_gc_mode"] for r in diagnosed)
    cat_freq = Counter(r["top_gc_category"] for r in diagnosed)

    print("\n" + "=" * 70)
    print("GEOCERT TOP-1 MODE FREQUENCY (all 61 traces)")
    print("=" * 70)
    print(f"  {'Mode':>10}  {'Name':<35}  {'N':>4}  {'%':>6}")
    print("  " + "-" * 60)
    for mode, n in mode_freq.most_common():
        # Find the name from any diagnosed record
        name = next(r["top_gc_name"] for r in diagnosed if r["top_gc_mode"] == mode)
        print(f"  {mode:>10}  {name:<35}  {n:>4}  {n/len(diagnosed)*100:>5.1f}%")

    print(f"\n  Category breakdown:")
    for cat in GC_CATS:
        n = cat_freq.get(cat, 0)
        print(f"    {GC_CAT_NAMES[cat]}: {n} ({n/len(diagnosed)*100:.0f}%)")

    # -----------------------------------------------------------------------
    # 5. Source breakdown (AG2 vs HyperAgent)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIAGNOSIS BY SOURCE SYSTEM")
    print("=" * 70)
    for src in ["AG2", "HyperAgent"]:
        src_diag = [r for r in diagnosed if r["source"] == src]
        src_mode_freq = Counter(r["top_gc_mode"] for r in src_diag)
        confs = [r["top_gc_confidence"] for r in src_diag]
        print(f"\n  {src} (n={len(src_diag)}):")
        print(f"    Mean confidence: {np.mean(confs):.3f}")
        print(f"    Mode distribution: " +
              "  ".join(f"{m}={n}" for m, n in src_mode_freq.most_common(3)))

    # -----------------------------------------------------------------------
    # 6. Save to results/mast_diagnosis.json
    # -----------------------------------------------------------------------
    output = {
        "meta": {
            "n_traces": len(diagnosed),
            "sources": {
                "AG2": sum(1 for r in diagnosed if r["source"] == "AG2"),
                "HyperAgent": sum(1 for r in diagnosed if r["source"] == "HyperAgent"),
            },
            "gc_mode_frequency": dict(mode_freq.most_common()),
            "gc_category_frequency": dict(cat_freq.most_common()),
            "mean_confidence": float(np.mean([r["top_gc_confidence"] for r in diagnosed])),
        },
        "cross_tabulation": xtab,
        "interpretation": interpretation,
        "traces": diagnosed,
    }

    out_path = RESULTS_DIR / "mast_diagnosis.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print(f"\n\nSaved: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
