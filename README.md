# RSCT-MAST: Structured Compatibility Certificates vs. Multi-Agent Failure Taxonomies

**Can structured certificates discover failure modes that binary annotation cannot?**

This repository compares two approaches to diagnosing multi-agent system failures:

- **[MAST](https://github.com/multi-agent-systems-failure-taxonomy/MAST)** (Cemri et al., 2025): A human-annotated taxonomy of 22 binary failure-mode labels across 61 multi-agent traces.
- **GeoCert**: A structured compatibility certification framework that produces continuous, multi-axis certificates per (solver x target) pair.

## Key Finding

MAST's three failure categories (Specification, Misalignment, Verification) are **well-separated** in structural signal space — z-score distances up to 3.155 between categories. But the certificate space is fundamentally higher-dimensional than binary labeling: PCA reveals 3+ orthogonal pathology axes that exist without any labels.

| Metric | MAST | GeoCert |
|--------|------|---------|
| Annotations per trace | 22 binary | Continuous n-attribute certificate |
| Failure categories | 3 (behavioral cause) | 3 x 3 = 9 (evaluation consequence) |
| Dimensionality | Fixed labels | Unbounded certificate axes |
| Discovery mode | Human annotation | Structural measurement |

## GeoCert Evaluation-Failure Taxonomy

Nine fine-grained failure modes organized into three categories:

**GC1 — Label Construction Failures**
- GCF-1.1: Tercile Uniformity
- GCF-1.2: Label-Solver Coupling
- GCF-1.3: Target-Difficulty Conflation

**GC2 — Decomposition Reduction Failures**
- GCF-2.1: Scalar Projection
- GCF-2.2: Gate Compression
- GCF-2.3: Range Compression

**GC3 — Deployment Translation Failures**
- GCF-3.1: Target-Solver Conflation
- GCF-3.2: Proxy Calibration Drift
- GCF-3.3: Fine-Routing Failure

## Quick Start

```bash
# Clone with MAST data
git clone --recursive https://github.com/NextShiftConsulting/rsct-mast.git
cd rsct-mast

# Or add MAST submodule separately
git submodule update --init

# Run the MAST signal analysis
pip install numpy
python analyze.py

# Run the GeoCert stress suite (injection-detection validation)
python run_s035.py
```

## Repository Structure

```
rsct-mast/
├── analyze.py          # MAST trace analysis: signal extraction + category separation
├── load_mast.py        # Load + normalize 61 annotated traces from MAST repo
├── signals.py          # 17 structural signal extractors (no labels, no classification)
├── stress_geocert.py   # GeoCert failure taxonomy: 9 modes, injection, diagnosis
├── run_s035.py         # Full stress suite runner
├── mast_repo/          # MAST dataset (git submodule)
├── PATENT_NOTICE.md    # Patent status
└── LICENSE             # Apache 2.0
```

## Results

### MAST Category Separation

| Pair | Z-Score Distance |
|------|-----------------|
| FC1_Specification vs FC2_Misalignment | **3.155** |
| FC2_Misalignment vs FC3_Verification | 1.862 |
| FC1_Specification vs FC3_Verification | 1.380 |

### Source Comparison (AG2 vs HyperAgent)

| Source | Traces | Mean Failures | Repetition | Handoff | Verify |
|--------|--------|---------------|------------|---------|--------|
| AG2 | 31 | 1.6 | 0.02 | 1.00 | 0.42 |
| HyperAgent | 30 | 2.1 | 0.67 | 0.00 | 0.06 |

AG2 and HyperAgent have radically different trajectory structures — AG2 is multi-turn dialogue (high handoff), HyperAgent is single-agent log streams (high repetition, zero handoff).

## Related Work

- **MAST**: Cemri et al., "Why Do Multi-Agent LLM Systems Fail?", [arXiv:2503.13657](https://arxiv.org/abs/2503.13657)
- **RSCT**: Martin, "Structured Compatibility Certification for Representation-Solver Systems", US Patent Application 19/575,615

## License

Apache 2.0 — see [LICENSE](LICENSE) and [PATENT_NOTICE.md](PATENT_NOTICE.md).
