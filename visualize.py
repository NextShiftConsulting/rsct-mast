"""
visualize.py — Publication-quality figures for MAST x GeoCert analysis.

Generates 4 figures and saves them to figures/:
  fig1_signal_heatmap.png     — Signal profiles by MAST category (z-score heatmap)
  fig2_pca_scatter.png        — PCA of 61 traces colored by MAST category
  fig3_category_separation.png — Pairwise category centroid distances
  fig4_geocert_taxonomy_grid.png — GeoCert failure mode expected signal patterns

Usage:
    python visualize.py
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for standalone use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import zscore

from load_mast import load_all_traces, MAST_CATEGORIES, get_active_categories
from signals import extract_signals
from stress_geocert import GEOCERT_FAILURE_SPECS, GeoCertFailureMode

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Signal keys to use (exclude incompatible-scale signals)
EXCLUDED_SIGNALS = {"n_steps", "total_chars", "mean_step_length"}

# Category display labels (short)
CATEGORY_LABELS = {
    "FC1_Specification": "FC1",
    "FC2_Misalignment": "FC2",
    "FC3_Verification": "FC3",
    "baseline": "baseline",
}

# Colors for categories (colorblind-safe)
CATEGORY_COLORS = {
    "FC1_Specification": "#E15759",   # red
    "FC2_Misalignment": "#4E79A7",    # blue
    "FC3_Verification": "#F28E2B",    # orange
    "baseline": "#59A14F",            # green
}

# Source marker shapes
SOURCE_MARKERS = {
    "AG2": "o",        # circle
    "HyperAgent": "^", # triangle
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load traces, extract signals, and group by MAST category."""
    print("Loading MAST traces...")
    traces = load_all_traces()
    print(f"  {len(traces)} traces loaded")

    print("Extracting signals...")
    for trace in traces:
        trace["signals"] = extract_signals(trace)

    # Determine signal keys (all except excluded, in deterministic order)
    all_keys = list(traces[0]["signals"].keys())
    signal_keys = [k for k in all_keys if k not in EXCLUDED_SIGNALS]

    # Group traces by MAST category (a trace can appear in multiple groups)
    cat_groups: dict[str, list] = defaultdict(list)
    for t in traces:
        cats = get_active_categories(t["annotations"])
        for c in cats:
            cat_groups[c].append(t)
        if not cats:
            cat_groups["baseline"].append(t)

    # Build signal matrix: shape (n_traces, n_signals)
    signal_matrix = np.array([
        [t["signals"][k] for k in signal_keys]
        for t in traces
    ])

    return traces, signal_keys, cat_groups, signal_matrix


# ---------------------------------------------------------------------------
# Figure 1: Signal Heatmap by MAST Category
# ---------------------------------------------------------------------------

def fig1_signal_heatmap(traces, signal_keys, cat_groups, signal_matrix):
    """Z-scored mean signal per MAST category x signal key."""
    print("Generating Figure 1: Signal Heatmap...")

    ordered_cats = ["FC1_Specification", "FC2_Misalignment", "FC3_Verification", "baseline"]
    present_cats = [c for c in ordered_cats if cat_groups[c]]

    # Compute global z-score normalization across all traces
    means = signal_matrix.mean(axis=0)
    stds = signal_matrix.std(axis=0)
    stds[stds == 0] = 1.0

    # Compute z-scored centroid for each category
    heatmap_data = []
    row_labels = []
    row_counts = []
    for cat in present_cats:
        group = cat_groups[cat]
        raw = np.array([[t["signals"][k] for k in signal_keys] for t in group])
        z_centroid = ((raw - means) / stds).mean(axis=0)
        heatmap_data.append(z_centroid)
        row_labels.append(CATEGORY_LABELS[cat])
        row_counts.append(len(group))

    heatmap_data = np.array(heatmap_data)  # shape: (n_cats, n_signals)

    # Readable column labels
    col_labels = [k.replace("_", "\n") for k in signal_keys]

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)

    vmax = max(abs(heatmap_data.max()), abs(heatmap_data.min()))
    vmax = max(vmax, 0.5)  # ensure non-trivial range

    im = ax.imshow(
        heatmap_data,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
    )

    # Annotate cells
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            val = heatmap_data[i, j]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=8, color=text_color, fontweight="bold",
            )

    # Axes
    ax.set_xticks(range(len(signal_keys)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=0)
    ax.set_yticks(range(len(present_cats)))
    y_labels = [f"{CATEGORY_LABELS[c]}\n(n={row_counts[i]})"
                for i, c in enumerate(present_cats)]
    ax.set_yticklabels(y_labels, fontsize=11)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("z-score", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(
        "Structural Signal Profiles by MAST Failure Category",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Signal", fontsize=11)
    ax.set_ylabel("MAST Category", fontsize=11)

    # Color y-tick labels to match category colors
    for tick, cat in zip(ax.get_yticklabels(), present_cats):
        tick.set_color(CATEGORY_COLORS[cat])

    out = FIGURES_DIR / "fig1_signal_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2: PCA Scatter of Traces
# ---------------------------------------------------------------------------

def fig2_pca_scatter(traces, signal_keys, signal_matrix):
    """PCA on signal matrix, colored by MAST category."""
    print("Generating Figure 2: PCA Scatter...")

    # Assign primary category label per trace (for coloring)
    # Priority: FC3 > FC2 > FC1 > baseline (most common first)
    priority = ["FC3_Verification", "FC2_Misalignment", "FC1_Specification", "baseline"]

    trace_cats = []
    for t in traces:
        cats = get_active_categories(t["annotations"])
        assigned = "baseline"
        for cat in priority:
            if cat in cats:
                assigned = cat
                break
        trace_cats.append(assigned)

    # Normalize
    means = signal_matrix.mean(axis=0)
    stds = signal_matrix.std(axis=0)
    stds[stds == 0] = 1.0
    X = (signal_matrix - means) / stds

    # PCA (manual — avoid sklearn dependency)
    cov = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project onto top 2 PCs
    pcs = X @ eigenvectors[:, :2]
    total_var = eigenvalues.sum()
    var_explained = eigenvalues[:2] / total_var * 100

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    # Plot each trace
    for i, (t, cat) in enumerate(zip(traces, trace_cats)):
        color = CATEGORY_COLORS[cat]
        marker = SOURCE_MARKERS[t["source"]]
        ax.scatter(
            pcs[i, 0], pcs[i, 1],
            c=color, marker=marker,
            s=80, alpha=0.75, linewidths=0.5, edgecolors="white",
            zorder=3,
        )

    # Legend — categories
    cat_patches = [
        mpatches.Patch(color=CATEGORY_COLORS[c], label=CATEGORY_LABELS[c])
        for c in ["FC1_Specification", "FC2_Misalignment", "FC3_Verification", "baseline"]
    ]
    # Legend — sources
    source_handles = [
        plt.Line2D([0], [0], marker="o", color="gray", linestyle="None",
                   markersize=8, label="AG2"),
        plt.Line2D([0], [0], marker="^", color="gray", linestyle="None",
                   markersize=8, label="HyperAgent"),
    ]
    legend1 = ax.legend(
        handles=cat_patches,
        title="MAST Category",
        loc="upper right",
        fontsize=10,
        title_fontsize=10,
        framealpha=0.85,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=source_handles,
        title="Source",
        loc="lower right",
        fontsize=10,
        title_fontsize=10,
        framealpha=0.85,
    )

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% var)", fontsize=12)
    ax.set_title("PCA of Multi-Agent Trace Signals", fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.axhline(0, color="lightgray", linewidth=0.8, zorder=1)
    ax.axvline(0, color="lightgray", linewidth=0.8, zorder=1)
    ax.grid(True, alpha=0.25, zorder=0)

    out = FIGURES_DIR / "fig2_pca_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Category Separation Bar Chart
# ---------------------------------------------------------------------------

def fig3_category_separation(traces, signal_keys, cat_groups, signal_matrix):
    """Horizontal bar chart of pairwise category centroid distances."""
    print("Generating Figure 3: Category Separation...")

    ordered_cats = ["FC1_Specification", "FC2_Misalignment", "FC3_Verification", "baseline"]
    present_cats = [c for c in ordered_cats if len(cat_groups[c]) >= 2]

    # Normalize
    means = signal_matrix.mean(axis=0)
    stds = signal_matrix.std(axis=0)
    stds[stds == 0] = 1.0

    # Z-scored centroids per category
    centroids = {}
    for cat in present_cats:
        group = cat_groups[cat]
        raw = np.array([[t["signals"][k] for k in signal_keys] for t in group])
        centroids[cat] = ((raw - means) / stds).mean(axis=0)

    # All pairwise distances
    pairs = []
    for c1, c2 in combinations(present_cats, 2):
        d = float(np.linalg.norm(centroids[c1] - centroids[c2]))
        label1 = CATEGORY_LABELS[c1]
        label2 = CATEGORY_LABELS[c2]
        pairs.append((f"{label1} vs {label2}", d))

    # Sort by distance descending
    pairs.sort(key=lambda x: x[1], reverse=True)

    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    # Color bars by magnitude
    bar_colors = []
    for v in values:
        if v >= 2.0:
            bar_colors.append("#59A14F")   # green
        elif v >= 1.0:
            bar_colors.append("#F28E2B")   # yellow/orange
        else:
            bar_colors.append("#E15759")   # red

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    y_pos = range(len(labels))
    bars = ax.barh(list(y_pos), values, color=bar_colors, edgecolor="white",
                   linewidth=0.8, height=0.6)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.03, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", ha="left", fontsize=11, fontweight="bold",
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Centroid Distance (z-score units)", fontsize=12)
    ax.set_title(
        "Pairwise Category Separation (z-score distance)",
        fontsize=14, fontweight="bold",
    )

    # Threshold reference lines
    ax.axvline(2.0, color="#59A14F", linestyle="--", linewidth=1.2,
               label="Strong (≥ 2.0)", alpha=0.8)
    ax.axvline(1.0, color="#F28E2B", linestyle="--", linewidth=1.2,
               label="Moderate (≥ 1.0)", alpha=0.8)

    ax.legend(fontsize=10, loc="lower right")
    ax.set_xlim(0, max(values) * 1.2)
    ax.tick_params(labelsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()  # largest on top

    out = FIGURES_DIR / "fig3_category_separation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 4: GeoCert Taxonomy Signal Pattern Grid
# ---------------------------------------------------------------------------

def fig4_geocert_taxonomy_grid():
    """Heatmap of GeoCert failure mode expected signal range centers."""
    print("Generating Figure 4: GeoCert Taxonomy Grid...")

    # Collect all signal keys across all modes (union)
    all_signal_keys: set[str] = set()
    for spec in GEOCERT_FAILURE_SPECS.values():
        all_signal_keys.update(spec.expected_signal_pattern.keys())
    col_keys = sorted(all_signal_keys)

    # Row order: GCF-1.1 through GCF-3.3 in enum order
    modes = list(GeoCertFailureMode)

    # Build matrix of range centers (NaN for unused signals)
    data = np.full((len(modes), len(col_keys)), np.nan)
    for i, mode in enumerate(modes):
        spec = GEOCERT_FAILURE_SPECS[mode]
        for j, key in enumerate(col_keys):
            if key in spec.expected_signal_pattern:
                lo, hi = spec.expected_signal_pattern[key]
                data[i, j] = (lo + hi) / 2.0

    # Row labels: "GCF-1.1\nTercile Uniformity"
    row_labels = []
    cat_band_colors = {
        "GC1": "#4E79A7",
        "GC2": "#F28E2B",
        "GC3": "#E15759",
    }
    row_cat = []
    for mode in modes:
        spec = GEOCERT_FAILURE_SPECS[mode]
        row_labels.append(f"{mode.value}\n{spec.name}")
        row_cat.append(spec.category.value)

    col_labels = [k.replace("_", "\n") for k in col_keys]

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)

    # Custom colormap with gray for NaN
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#E8E8E8")

    im = ax.imshow(
        data,
        cmap=cmap,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )

    # Annotate non-NaN cells with range center value
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.7 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold",
                )
            else:
                ax.text(
                    j, i, "—",
                    ha="center", va="center",
                    fontsize=8, color="#AAAAAA",
                )

    # Axes
    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=0)
    ax.set_yticks(range(len(modes)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Color y-tick labels by GeoCert category
    for tick, cat in zip(ax.get_yticklabels(), row_cat):
        tick.set_color(cat_band_colors[cat])

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Expected range center [0, 1]", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Category legend
    cat_patches = [
        mpatches.Patch(color=cat_band_colors["GC1"], label="GC1: Label Construction"),
        mpatches.Patch(color=cat_band_colors["GC2"], label="GC2: Decomposition/Reduction"),
        mpatches.Patch(color=cat_band_colors["GC3"], label="GC3: Deployment/Translation"),
    ]
    ax.legend(
        handles=cat_patches,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.06),
        ncol=3,
        fontsize=9,
        framealpha=0.9,
        title="GeoCert Category",
        title_fontsize=9,
    )

    # Horizontal separators between GC stages (after row 2 and row 5)
    for sep_y in [2.5, 5.5]:
        ax.axhline(sep_y, color="white", linewidth=2.5)

    ax.set_title(
        "GeoCert Failure Mode Expected Signal Patterns",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Signal Key", fontsize=11)
    ax.set_ylabel("GeoCert Failure Mode", fontsize=11)

    out = FIGURES_DIR / "fig4_geocert_taxonomy_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MAST x GeoCert Visualization")
    print("=" * 60)

    traces, signal_keys, cat_groups, signal_matrix = load_data()

    print(f"\nSignal keys used ({len(signal_keys)}): {signal_keys}")
    print(f"Category sizes:")
    for cat in ["FC1_Specification", "FC2_Misalignment", "FC3_Verification", "baseline"]:
        print(f"  {cat}: {len(cat_groups[cat])} traces")

    print()
    fig1_signal_heatmap(traces, signal_keys, cat_groups, signal_matrix)
    fig2_pca_scatter(traces, signal_keys, signal_matrix)
    fig3_category_separation(traces, signal_keys, cat_groups, signal_matrix)
    fig4_geocert_taxonomy_grid()

    print("\nAll figures saved to:", FIGURES_DIR.resolve())
    print("Done.")


if __name__ == "__main__":
    main()
