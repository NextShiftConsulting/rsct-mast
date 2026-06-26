"""
diagnose.py — User-facing API for GeoCert diagnosis of multi-agent traces.

Lets users bring their own multi-agent trajectory data and receive a
structured GeoCert evaluation-failure diagnosis. Works as both a library
and a CLI tool.

Library usage:
    from diagnose import diagnose_trace
    result = diagnose_trace(trajectory, source="AG2")

CLI usage:
    python diagnose.py trajectory.json
    cat trajectory.json | python diagnose.py -
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make sure local modules are importable when run from any directory
sys.path.insert(0, str(Path(__file__).parent))

from signals import extract_signals
from stress_geocert import diagnose_geocert_failure


# ---------------------------------------------------------------------------
# Signal mapping
# ---------------------------------------------------------------------------

def map_trace_signals_to_geocert(trace_signals: dict) -> dict:
    """
    Map trajectory-level structural signals to GeoCert diagnostic signal space.

    The trace signals come from extract_signals() and measure execution
    behaviour (repetition, handoffs, error density, etc.).  The GeoCert
    signal space measures evaluation-failure properties (kappa compression,
    gate entropy, etc.).  This function bridges the two spaces with
    heuristic mappings grounded in what each signal pair shares semantically.

    Mapping rationale (signal → GeoCert key):
    ──────────────────────────────────────────
    kappa_compression
        High repetition_ratio + consecutive_dupe_rate → compressed, loopy
        behaviour.  A trace stuck repeating itself behaves like a certificate
        space that has been compressed onto a single point.  Proxy: average of
        both repetition measures.

    gate_entropy
        verify_density → fraction of steps that contain checking/validation
        language.  Low verification = the agent never "gates" its outputs = low
        gate entropy in the certificate sense.  Proxy: min(verify_density / 3,
        1.0) — saturates at ~3 verify tokens per step.

    aggregate_separation
        handoff_rate → how often control passes between agents.  Frequent
        handoffs indicate distinct agent roles, which parallels good aggregate
        separation between certificate profiles.  Proxy: handoff_rate.

    solver_label_dependence
        dominance → fraction of steps owned by the most active agent.  High
        dominance means one "solver" drives the whole outcome, analogous to
        labels that are dominated by a single solver's behaviour.
        Proxy: dominance.

    false_execute_risk
        error_density → errors-per-step.  Traces with many explicit error
        markers carry a higher risk of the system executing despite known
        failure conditions.  Proxy: min(error_density / 5, 1.0) — saturates
        at ~5 error tokens per step.

    fine_routing_accuracy
        code_density + entropy_agents together proxy for structured, diverse
        routing.  High code density means the agents are producing concrete
        outputs (good routing), high entropy_agents means routing is spread
        across multiple agents (also good).  Proxy: min((code_density/10 +
        entropy_agents/3) / 2, 1.0).

    delta_structure
        growth_ratio → whether the second half of the trajectory is larger
        than the first.  A trajectory that expands (growth_ratio > 1) has
        increasing structure (positive delta); one that shrinks has negative
        delta.  Proxy: min(growth_ratio / 2, 1.0).

    sigma_outlier
        length_cv → coefficient of variation of step lengths.  High CV means
        some steps are wildly longer than others, analogous to a high-sigma
        outlier in certificate distributions.  Proxy: min(length_cv, 1.0).

    label_entropy
        entropy_agents → Shannon entropy over agent participation.  Many
        active agents with equal voice = high label entropy.  Proxy:
        min(entropy_agents / 3, 1.0) — normalised by log2(8) ≈ 3 bits.

    prediction_entropy
        max_role_bigram → how structured the role sequence is.  A very
        regular role sequence (high max_role_bigram) predicts a low-entropy
        system; an irregular one predicts high entropy.  Proxy: 1 -
        max_role_bigram.

    noisy_control_similarity
        repetition_ratio alone (without consecutive_dupe_rate) proxies for
        how similar the trace is to a noisy baseline that just replays steps.
        Proxy: repetition_ratio.

    trf_correlation
        tail_termination → whether the trace reaches a clear conclusion.
        Traces that end definitively correlate with task-recoverability
        ceiling (TRF).  Proxy: tail_termination (already in [0,1]).

    target_variance
        n_steps normalised to [0,1].  Longer traces tend to cover more
        diverse targets/scenarios, increasing effective target variance.
        Proxy: min(n_steps / 100, 1.0).

    cross_solver_label_agreement
        1 - dominance → low dominance means many agents participate equally,
        analogous to high cross-solver agreement on a label.
        Proxy: 1 - dominance.

    solver_specificity
        n_agents normalised.  More distinct agents = more specialised roles =
        higher solver specificity.  Proxy: min(n_agents / 5, 1.0).

    alpha_rank_gap
        error_density (separate from false_execute_risk): traces with many
        errors tend to have large gaps between what an idealised ranking
        (alpha) would say and what actually happened.
        Proxy: min(error_density / 3, 1.0).

    alpha_profile_corr
        handoff_rate and verify_density together: frequent handoffs with low
        verification suggest alpha profiles that are uncorrelated across
        agents.  Proxy: max(0.0, 1.0 - handoff_rate - verify_density/5).

    metric_range
        Combination of total_chars and n_steps variance (length_cv).  A wide
        range of step lengths and total content proxies for a wide metric
        range.  Proxy: min((length_cv + min(total_chars / 50000, 1.0)) / 2,
        1.0).

    all_execute_rate
        has_termination (binary).  A trace with a clean termination signal is
        more likely to have "executed" to completion.
        Proxy: has_termination.

    native_tightness
        1 - length_cv.  Low variability in step lengths → tightly clustered
        native behaviour.  Proxy: max(0.0, 1.0 - length_cv).

    proxy_success_corr
        verify_density and has_termination combined.  High verification AND a
        clean termination are jointly necessary for proxy success to correlate
        with actual success.  Proxy: min((verify_density/5 + has_termination)
        / 2, 1.0).

    serious_solver_range
        entropy_agents normalised.  A wider entropy over agents means the
        serious contributors span a wider range.  Proxy:
        min(entropy_agents / 3, 1.0).

    Args:
        trace_signals: Output of extract_signals() — 17 raw structural
            measurements from the trajectory.

    Returns:
        Dict with all 22 GeoCert signal keys mapped to [0, 1] floats.
    """
    r = trace_signals.get("repetition_ratio", 0.0)
    cdr = trace_signals.get("consecutive_dupe_rate", 0.0)
    vd = trace_signals.get("verify_density", 0.0)
    hr = trace_signals.get("handoff_rate", 0.0)
    dom = trace_signals.get("dominance", 1.0)
    ed = trace_signals.get("error_density", 0.0)
    cd = trace_signals.get("code_density", 0.0)
    ea = trace_signals.get("entropy_agents", 0.0)
    gr = trace_signals.get("growth_ratio", 1.0)
    lcv = trace_signals.get("length_cv", 0.0)
    na = trace_signals.get("n_agents", 1.0)
    ns = trace_signals.get("n_steps", 0.0)
    tc = trace_signals.get("total_chars", 0.0)
    ht = trace_signals.get("has_termination", 0.0)
    tt = trace_signals.get("tail_termination", 0.0)
    mrb = trace_signals.get("max_role_bigram", 0.0)

    def clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    return {
        # Repetition → compressed, loopy behaviour
        "kappa_compression": clamp((r + cdr) / 2.0),

        # Verification density → gate entropy
        "gate_entropy": clamp(vd / 3.0),

        # Handoff rate → aggregate separation between agents
        "aggregate_separation": clamp(hr),

        # Dominance → label/outcome controlled by one agent-solver
        "solver_label_dependence": clamp(dom),

        # Error density → risk of executing despite failure
        "false_execute_risk": clamp(ed / 5.0),

        # Code density + agent entropy → fine-grained routing ability
        "fine_routing_accuracy": clamp((cd / 10.0 + ea / 3.0) / 2.0),

        # Growth ratio → trajectory has increasing structure
        "delta_structure": clamp(gr / 2.0),

        # Length CV → outlier turbulence in step sizes
        "sigma_outlier": clamp(lcv),

        # Agent entropy → diversity of label/output sources
        "label_entropy": clamp(ea / 3.0),

        # Inverse of role regularity → unpredictable output distribution
        "prediction_entropy": clamp(1.0 - mrb),

        # Raw repetition ratio → similarity to a noisy baseline
        "noisy_control_similarity": clamp(r),

        # Tail termination → trace reaches a clear, recoverable conclusion
        "trf_correlation": clamp(tt),

        # Trace length → proxy for diversity of covered targets
        "target_variance": clamp(ns / 100.0),

        # Inverse dominance → multiple agents agree, like cross-solver agreement
        "cross_solver_label_agreement": clamp(1.0 - dom),

        # Number of distinct agents → role specialisation
        "solver_specificity": clamp(na / 5.0),

        # Error density → gap between idealised ranking and actual outcome
        "alpha_rank_gap": clamp(ed / 3.0),

        # High handoff with low verify → uncorrelated agent alpha profiles
        "alpha_profile_corr": clamp(max(0.0, 1.0 - hr - vd / 5.0)),

        # Length variance + content volume → wide metric range
        "metric_range": clamp((lcv + min(tc / 50000.0, 1.0)) / 2.0),

        # Clean termination → system reached an execute decision
        "all_execute_rate": clamp(ht),

        # Low length variability → tightly clustered native behaviour
        "native_tightness": clamp(1.0 - lcv),

        # Verification + termination → proxy success correlates with actual
        "proxy_success_corr": clamp((vd / 5.0 + ht) / 2.0),

        # Agent entropy → serious contributors span a wider range
        "serious_solver_range": clamp(ea / 3.0),
    }


# ---------------------------------------------------------------------------
# Core diagnosis function
# ---------------------------------------------------------------------------

def diagnose_trace(
    trajectory: List[Dict[str, Any]],
    source: str = "unknown",
) -> Dict[str, Any]:
    """
    Diagnose a multi-agent trace for GeoCert evaluation failures.

    Args:
        trajectory: List of step dicts, each with 'content' (str),
                    'role' (str), and optionally 'name' (str).
        source: Label for the agent system (e.g., "AG2", "CrewAI").

    Returns:
        Dict with keys:
            'source'    — the source label passed in
            'n_steps'   — number of trajectory steps
            'signals'   — raw structural signals from extract_signals()
            'geocert_signals' — mapped GeoCert signal values
            'diagnosis' — output of diagnose_geocert_failure() with top1_mode,
                          candidates (mode, name, category, confidence),
                          and input_signals
            'summary'   — human-readable one-line description of top diagnosis
    """
    # Normalise: ensure every step has the required keys
    normalised: List[Dict[str, str]] = []
    for step in trajectory:
        content = step.get("content", "")
        if isinstance(content, list):
            content = "\n".join(str(x) for x in content)
        normalised.append({
            "content": str(content),
            "role": str(step.get("role", "unknown")),
            "name": str(step.get("name", "")),
        })

    # Wrap into the trace format that extract_signals expects
    trace = {
        "trajectory": normalised,
        "source": source,
        "n_steps": len(normalised),
    }

    # Extract raw structural signals
    raw_signals = extract_signals(trace)

    # Map to GeoCert signal space
    geocert_signals = map_trace_signals_to_geocert(raw_signals)

    # Run GeoCert diagnosis
    diagnosis = diagnose_geocert_failure(geocert_signals, top_k=3)

    # Build a readable summary
    top = diagnosis["candidates"][0]
    summary = (
        f"{top['mode']} ({top['name']}) — confidence {top['confidence']:.3f} "
        f"[{top['category']}]"
    )

    return {
        "source": source,
        "n_steps": len(normalised),
        "signals": raw_signals,
        "geocert_signals": geocert_signals,
        "diagnosis": diagnosis,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_trajectory_from_file(path: str) -> List[Dict[str, Any]]:
    """Load trajectory from a JSON file or stdin ('-')."""
    if path == "-":
        data = json.load(sys.stdin)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

    # Accept three shapes:
    #   1. A bare list of step dicts  →  use directly
    #   2. {"trajectory": [...]}      →  extract trajectory
    #   3. A dict with any other key that is a list  →  try first list value
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "trajectory" in data:
            return data["trajectory"]
        # Fallback: first list-valued key
        for v in data.values():
            if isinstance(v, list):
                return v
    raise ValueError(
        f"Cannot parse trajectory from {path!r}. "
        "Expected a JSON array or a dict with a 'trajectory' key."
    )


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnose a multi-agent trajectory for GeoCert evaluation failures."
    )
    parser.add_argument(
        "input",
        help="Path to a JSON file containing a trajectory, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--source",
        default="unknown",
        help="Label for the agent system (e.g. AG2, CrewAI). Default: 'unknown'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of candidate failure modes to show. Default: 3.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output full result as JSON instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    try:
        trajectory = _load_trajectory_from_file(args.input)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    result = diagnose_trace(trajectory, source=args.source)

    if args.output_json:
        print(json.dumps(result, indent=2))
        return

    # Human-readable output
    print(f"\nGeoCert Diagnosis")
    print(f"{'=' * 50}")
    print(f"Source  : {result['source']}")
    print(f"Steps   : {result['n_steps']}")
    print(f"Summary : {result['summary']}")
    print()
    print("Top candidates:")
    for i, c in enumerate(result["diagnosis"]["candidates"][: args.top_k], 1):
        print(f"  {i}. {c['mode']:8s}  {c['name']:<35}  conf={c['confidence']:.3f}  [{c['category']}]")
    print()
    print("Key structural signals:")
    sig_display = [
        ("repetition_ratio", "repetition"),
        ("consecutive_dupe_rate", "consec_dupe"),
        ("handoff_rate", "handoff"),
        ("verify_density", "verify"),
        ("error_density", "error"),
        ("dominance", "dominance"),
        ("entropy_agents", "agent_entropy"),
        ("length_cv", "length_cv"),
    ]
    for key, label in sig_display:
        v = result["signals"].get(key, 0.0)
        print(f"  {label:<18}: {v:.4f}")


if __name__ == "__main__":
    main()
