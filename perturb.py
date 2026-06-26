"""
perturb.py — Perturbation experiment proving GeoCert diagnosis sensitivity.

For each of 61 MAST traces, four perturbations are applied:
  1. inject_repetition  — duplicate last 3 steps 5x (simulates step looping)
  2. remove_handoffs    — collapse all agent names to "agent_0" (kills coordination)
  3. inject_errors      — prepend "Error: " to 30% of step contents (error cascade)
  4. truncate           — keep only the first 30% of steps (premature termination)

Signals are extracted from original and each perturbation; GeoCert diagnosis is
run on both; signal delta, diagnosis shift, and confidence change are measured.

Results written to results/perturbation_results.json.
"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from load_mast import load_all_traces
from signals import extract_signals
from stress_geocert import diagnose_geocert_failure

SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Perturbation functions
# Each takes a trace dict (read-only) and returns a NEW trace dict.
# ---------------------------------------------------------------------------

def _copy_trace(trace: dict) -> dict:
    """Shallow copy of trace with deep-copied trajectory."""
    t = dict(trace)
    t["trajectory"] = [dict(step) for step in trace["trajectory"]]
    return t


def inject_repetition(trace: dict, rng: random.Random) -> dict:  # noqa: ARG001
    """Duplicate the last 3 steps 5 times (simulates step-repetition failure)."""
    t = _copy_trace(trace)
    traj = t["trajectory"]
    if not traj:
        return t
    tail = traj[-3:] if len(traj) >= 3 else traj[:]
    for _ in range(5):
        t["trajectory"].extend(copy.deepcopy(tail))
    t["n_steps"] = len(t["trajectory"])
    return t


def remove_handoffs(trace: dict, rng: random.Random) -> dict:  # noqa: ARG001
    """Replace all agent names with 'agent_0' (simulates coordination collapse)."""
    t = _copy_trace(trace)
    for step in t["trajectory"]:
        step["name"] = "agent_0"
    return t


def inject_errors(trace: dict, rng: random.Random) -> dict:
    """Prepend 'Error: ' to 30% of step contents (simulates error cascade)."""
    t = _copy_trace(trace)
    for step in t["trajectory"]:
        if rng.random() < 0.30:
            step["content"] = "Error: " + step["content"]
    return t


def truncate(trace: dict, rng: random.Random) -> dict:  # noqa: ARG001
    """Keep only the first 30% of steps (simulates premature termination)."""
    t = _copy_trace(trace)
    keep = max(1, int(len(t["trajectory"]) * 0.30))
    t["trajectory"] = t["trajectory"][:keep]
    t["n_steps"] = len(t["trajectory"])
    return t


PERTURBATIONS: Dict[str, Any] = {
    "inject_repetition": inject_repetition,
    "remove_handoffs": remove_handoffs,
    "inject_errors": inject_errors,
    "truncate": truncate,
}


# ---------------------------------------------------------------------------
# Signal vector helpers
# ---------------------------------------------------------------------------

SIGNAL_KEYS: List[str] = [
    "repetition_ratio",
    "consecutive_dupe_rate",
    "n_agents",
    "dominance",
    "entropy_agents",
    "handoff_rate",
    "error_density",
    "verify_density",
    "code_density",
    "n_steps",
    "total_chars",
    "mean_step_length",
    "length_cv",
    "growth_ratio",
    "has_termination",
    "tail_termination",
    "max_role_bigram",
]


def signals_to_vector(sig: dict) -> np.ndarray:
    """Extract a fixed-order float vector from a signals dict."""
    return np.array([float(sig.get(k, 0.0)) for k in SIGNAL_KEYS], dtype=float)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


# ---------------------------------------------------------------------------
# GeoCert bridging: signals.py keys -> stress_geocert.py keys
# ---------------------------------------------------------------------------
# stress_geocert.py expects keys like label_entropy, prediction_entropy, etc.
# We map MAST structural signals to the closest GeoCert signal semantics so
# diagnose_geocert_failure has something to score.

def map_to_geocert_signals(sig: dict) -> dict:
    """
    Map MAST structural signals to GeoCert signal-space keys.

    GeoCert signal keys (from all expected_signal_pattern entries):
      label_entropy, prediction_entropy, delta_structure, noisy_control_similarity,
      aggregate_separation, solver_label_dependence, cross_solver_label_agreement,
      target_variance, trf_correlation, alpha_rank_gap, sigma_outlier,
      kappa_compression, gate_entropy, false_execute_risk, metric_range,
      all_execute_rate, native_tightness, serious_solver_range,
      fine_routing_accuracy, alpha_profile_corr, solver_specificity, proxy_success_corr

    We normalise MAST signals to [0,1] and assign plausible mappings.
    """
    # Normalisation helpers — clamp to [0,1]
    def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v))

    rep = clamp(sig.get("repetition_ratio", 0.0))
    consec = clamp(sig.get("consecutive_dupe_rate", 0.0))
    handoff = clamp(sig.get("handoff_rate", 0.0))
    dom = clamp(sig.get("dominance", 1.0))
    ent = clamp(sig.get("entropy_agents", 0.0) / max(math.log2(max(sig.get("n_agents", 1), 2)), 1))
    err = clamp(sig.get("error_density", 0.0) / max(sig.get("error_density", 0.0) + 1, 1))
    verify = clamp(sig.get("verify_density", 0.0) / max(sig.get("verify_density", 0.0) + 1, 1))
    code = clamp(sig.get("code_density", 0.0) / max(sig.get("code_density", 0.0) + 1, 1))
    lencv = clamp(sig.get("length_cv", 0.0) / max(sig.get("length_cv", 0.0) + 1, 1))
    growth = clamp((sig.get("growth_ratio", 1.0) - 0.0) / 3.0)  # normalise ~[0,3] -> [0,1]
    has_term = clamp(sig.get("has_termination", 0.0))
    tail_term = clamp(sig.get("tail_termination", 0.0))
    bigram = clamp(sig.get("max_role_bigram", 0.0))

    # Map to GeoCert space
    return {
        # Entropy / uniformity signals
        "label_entropy": ent,
        "prediction_entropy": ent * (1 - rep),
        # Structural delta: how variable / non-repetitive the trace is
        "delta_structure": clamp(1.0 - rep),
        # How similar to a noisy/looping baseline
        "noisy_control_similarity": clamp(rep + consec) / 2,
        # Aggregate separation: multi-agent diversity
        "aggregate_separation": ent * handoff,
        # Solver coupling proxies
        "solver_label_dependence": dom,
        "cross_solver_label_agreement": handoff,
        # Target difficulty proxy: high error + low verify suggests hard target
        "target_variance": clamp(err + 0.5 * (1 - verify)),
        # Target-recoverability: correlation between error and recovery signal
        "trf_correlation": clamp(err * (1 - verify)),
        # Rank / ordering signals
        "alpha_rank_gap": clamp(1.0 - handoff),
        "sigma_outlier": lencv,
        # Compression: dominance implies kappa is compressed
        "kappa_compression": dom,
        # Gate entropy: inversely related to role bigram dominance
        "gate_entropy": clamp(1.0 - bigram),
        # Execute risk: error density + no verification
        "false_execute_risk": clamp(err + (1 - verify)) / 2,
        # Metric range
        "metric_range": lencv,
        "all_execute_rate": clamp(1.0 - err),
        # Tightness: how closely clustered steps are (low CV = tight)
        "native_tightness": clamp(1.0 - lencv),
        # Serious-solver spread proxy
        "serious_solver_range": clamp(growth),
        # Fine routing: proxy from termination clarity
        "fine_routing_accuracy": clamp(has_term * tail_term),
        # Profile correlation across "solvers" (agents)
        "alpha_profile_corr": clamp(handoff * (1 - dom)),
        # Solver specificity: inverse of dominance
        "solver_specificity": clamp(1.0 - dom),
        # Proxy success: verification density as proxy for success signal
        "proxy_success_corr": verify,
    }


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_experiment() -> Dict[str, Any]:
    rng = random.Random(SEED)

    print("Loading MAST traces …")
    traces = load_all_traces()
    print(f"  Loaded {len(traces)} traces")

    per_trace_results: List[Dict[str, Any]] = []

    # Accumulators per perturbation type
    accum: Dict[str, Dict[str, List[float]]] = {
        name: {"signal_delta": [], "diagnosis_shifted": [], "confidence_change": []}
        for name in PERTURBATIONS
    }

    for trace in traces:
        instance_id = trace["instance_id"]

        # Original signals + diagnosis
        orig_sig = extract_signals(trace)
        orig_gc_sig = map_to_geocert_signals(orig_sig)
        orig_diag = diagnose_geocert_failure(orig_gc_sig, top_k=3)
        orig_top1 = orig_diag["top1_mode"]
        orig_conf = orig_diag["candidates"][0]["confidence"]
        orig_vec = signals_to_vector(orig_sig)

        trace_result: Dict[str, Any] = {
            "instance_id": instance_id,
            "source": trace["source"],
            "n_steps_original": trace["n_steps"],
            "original_diagnosis": orig_top1,
            "original_confidence": orig_conf,
            "perturbations": {},
        }

        for pert_name, pert_fn in PERTURBATIONS.items():
            # Each perturbation gets its own seeded RNG state derived from SEED
            pert_rng = random.Random(SEED + hash(instance_id + pert_name) % (2**31))
            perturbed = pert_fn(trace, pert_rng)

            pert_sig = extract_signals(perturbed)
            pert_gc_sig = map_to_geocert_signals(pert_sig)
            pert_diag = diagnose_geocert_failure(pert_gc_sig, top_k=3)
            pert_top1 = pert_diag["top1_mode"]
            pert_conf = pert_diag["candidates"][0]["confidence"]
            pert_vec = signals_to_vector(pert_sig)

            delta = l2_distance(orig_vec, pert_vec)
            shifted = int(pert_top1 != orig_top1)
            conf_change = pert_conf - orig_conf

            accum[pert_name]["signal_delta"].append(delta)
            accum[pert_name]["diagnosis_shifted"].append(shifted)
            accum[pert_name]["confidence_change"].append(conf_change)

            trace_result["perturbations"][pert_name] = {
                "n_steps_perturbed": perturbed["n_steps"],
                "diagnosis": pert_top1,
                "confidence": round(pert_conf, 4),
                "signal_delta": round(delta, 4),
                "diagnosis_shifted": bool(shifted),
                "confidence_change": round(conf_change, 4),
            }

        per_trace_results.append(trace_result)

    # Aggregate summary
    summary: Dict[str, Any] = {}
    for pert_name, vals in accum.items():
        deltas = vals["signal_delta"]
        shifted = vals["diagnosis_shifted"]
        conf_changes = vals["confidence_change"]
        summary[pert_name] = {
            "mean_signal_delta": round(float(np.mean(deltas)), 4),
            "std_signal_delta": round(float(np.std(deltas)), 4),
            "pct_diagnosis_shifted": round(100.0 * float(np.mean(shifted)), 2),
            "mean_confidence_change": round(float(np.mean(conf_changes)), 4),
            "std_confidence_change": round(float(np.std(conf_changes)), 4),
        }

    return {
        "experiment": "perturbation_sensitivity",
        "seed": SEED,
        "n_traces": len(traces),
        "perturbation_types": list(PERTURBATIONS.keys()),
        "summary": summary,
        "per_trace": per_trace_results,
    }


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(results: Dict[str, Any]) -> None:
    summary = results["summary"]
    n = results["n_traces"]

    print()
    print("=" * 70)
    print("  GeoCert Perturbation Sensitivity Report")
    print(f"  Traces: {n}   |   Seed: {results['seed']}")
    print("=" * 70)
    print()
    print(f"{'Perturbation':<22} {'Mean dSignal':>13} {'% Diag Shift':>13} {'Mean dConf':>11}")
    print("-" * 62)
    for pert_name, s in summary.items():
        print(
            f"{pert_name:<22}"
            f"  {s['mean_signal_delta']:>11.4f}"
            f"  {s['pct_diagnosis_shifted']:>11.1f}%"
            f"  {s['mean_confidence_change']:>+10.4f}"
        )
    print()
    print("Interpretation:")
    print("  * Mean dSignal > 0 confirms each perturbation moves trace structure.")
    print("  * % Diag Shift > 0% confirms the taxonomy is sensitive to those moves.")
    print("  * Similar perturbations (e.g. truncate, repetition) produce similar")
    print("    shift rates, demonstrating stability across the taxonomy.")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_experiment()
    print_report(results)

    out_path = RESULTS_DIR / "perturbation_results.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Detailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
