"""
dropletqcpy.empty_droplets
==========================

Identify empty droplets and low-quality bins using a two-dimensional
Gaussian Mixture Model (GMM) on log10(total_counts) and nuclear_fraction.

The decision is cluster-level, not cell-level:
  - Fit a 4-component GMM on [log10(total_counts+1), nuclear_fraction]
  - Clusters with mean_log10_total_counts < 2 OR mean_nuclear_fraction < 0.1
    are flagged as low-quality

Reference
---------
Muskovic & Kim (2021). DropletQC: improved identification of empty
droplets and damaged cells in single-cell RNA-seq data.
Genome Biology, 22, 329. https://doi.org/10.1186/s13059-021-02547-0
"""

from __future__ import annotations

import datetime
import logging
import os
import textwrap
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# OpenBLAS / OMP thread guard
# ---------------------------------------------------------------------------
# On machines with many cores (>128), OpenBLAS will segfault unless thread
# count is capped. Set conservative defaults here if not already set by user.
# These must be set before sklearn/numpy are initialised.
for _env_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if _env_var not in os.environ:
        os.environ[_env_var] = "4"
del _env_var

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_LABEL_CELL  = "cell"
_LABEL_EMPTY = "empty_droplet"

# Column names written to adata.obs by sc.pp.calculate_qc_metrics
_QC_TOTAL_COUNTS  = "total_counts"
_QC_N_GENES       = "n_genes_by_counts"
_QC_PCT_MITO      = "pct_counts_mt"


# ---------------------------------------------------------------------------
# Step 0 — QC metrics + feature construction
# ---------------------------------------------------------------------------

def _ensure_qc_metrics(
    adata: anndata.AnnData,
    total_counts_key: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure total_counts is available in adata.obs, running
    sc.pp.calculate_qc_metrics if necessary.

    Returns
    -------
    total_counts      : np.ndarray, shape (n_obs,)
    log10_total_counts: np.ndarray, shape (n_obs,)
    """
    key = total_counts_key or _QC_TOTAL_COUNTS

    if key not in adata.obs.columns:
        logger.info(
            "'%s' not found in adata.obs — running sc.pp.calculate_qc_metrics.", key
        )
        # Detect mitochondrial genes if not already flagged
        if "mt" not in adata.var.columns:
            adata.var["mt"] = adata.var_names.str.startswith(("mt-", "MT-"))
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], inplace=True, percent_top=None
        )
        logger.info(
            "QC metrics computed: total_counts, n_genes_by_counts, pct_counts_mt."
        )

    if key not in adata.obs.columns:
        # Fallback: compute raw sum directly
        logger.warning(
            "'%s' still missing after calculate_qc_metrics — "
            "computing from adata.X.sum(axis=1).", key
        )
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float64)
        adata.obs[key] = total_counts
    else:
        total_counts = adata.obs[key].to_numpy(dtype=np.float64)

    log10_total_counts = np.log10(total_counts + 1.0)
    return total_counts, log10_total_counts


# ---------------------------------------------------------------------------
# Step 1 — Hard filter
# ---------------------------------------------------------------------------

def _hard_filter(log10_tc: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask: True for bins passing log10(total_counts+1) > threshold."""
    return log10_tc > threshold


# ---------------------------------------------------------------------------
# Step 3b — Auto-detect n_components from NF distribution
# ---------------------------------------------------------------------------

def _detect_n_components(
    nf: np.ndarray,
    min_components: int = 2,
    max_components: int = 6,
    kde_bw: str = "scott",
    n_grid: int = 512,
) -> int:
    """Estimate n_components by counting KDE peaks in NF distribution."""
    from scipy.signal import argrelextrema
    from scipy.stats import gaussian_kde

    nf_clean = nf[np.isfinite(nf)]
    if len(nf_clean) < 50:
        logger.warning(
            "Too few finite NF values (%d) for peak detection; "
            "using n_components=%d.", len(nf_clean), min_components
        )
        return min_components

    kde = gaussian_kde(nf_clean, bw_method=kde_bw)
    grid = np.linspace(nf_clean.min(), nf_clean.max(), n_grid)
    density = kde(grid)

    from scipy.signal import argrelextrema
    peaks = argrelextrema(density, np.greater, order=10)[0]
    n_peaks = max(1, len(peaks))
    n_components = int(np.clip(n_peaks + 1, min_components, max_components))

    logger.info(
        "NF peak detection: %d peak(s) found -> n_components set to %d.",
        n_peaks, n_components,
    )
    return n_components


# ---------------------------------------------------------------------------
# Step 4 — Cluster-level statistics (extended)
# ---------------------------------------------------------------------------

def _cluster_stats(
    log10_tc: np.ndarray,
    nf: np.ndarray,
    total_counts: np.ndarray,
    labels: np.ndarray,
    n_components: int,
) -> pd.DataFrame:
    """
    Compute per-cluster summary statistics.

    Columns
    -------
    n_bins                  : number of bins in cluster
    mean_total_counts       : mean raw total counts
    median_total_counts     : median raw total counts
    mean_log10_total_counts : mean log10(total_counts+1)
    mean_nuclear_fraction   : mean NF
    median_nuclear_fraction : median NF
    min_nuclear_fraction    : min NF
    max_nuclear_fraction    : max NF
    """
    rows = []
    for k in range(n_components):
        mask = labels == k
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "cluster": k,
                "n_bins": 0,
                "mean_total_counts": np.nan,
                "median_total_counts": np.nan,
                "mean_log10_total_counts": np.nan,
                "mean_nuclear_fraction": np.nan,
                "median_nuclear_fraction": np.nan,
                "min_nuclear_fraction": np.nan,
                "max_nuclear_fraction": np.nan,
            })
        else:
            rows.append({
                "cluster": k,
                "n_bins": n,
                "mean_total_counts": float(np.mean(total_counts[mask])),
                "median_total_counts": float(np.median(total_counts[mask])),
                "mean_log10_total_counts": float(np.mean(log10_tc[mask])),
                "mean_nuclear_fraction": float(np.mean(nf[mask])),
                "median_nuclear_fraction": float(np.median(nf[mask])),
                "min_nuclear_fraction": float(np.min(nf[mask])),
                "max_nuclear_fraction": float(np.max(nf[mask])),
            })
    return pd.DataFrame(rows).set_index("cluster")


# ---------------------------------------------------------------------------
# Step 5 — Drop rule
# ---------------------------------------------------------------------------

def _flag_low_quality_clusters(
    stats: pd.DataFrame,
    min_total_counts: float,
    min_nuclear_fraction: float,
) -> pd.Index:
    """
    Flag clusters as low-quality if:
        mean_total_counts < min_total_counts
        OR mean_nuclear_fraction < min_nuclear_fraction
    """
    return stats[
        (stats["mean_total_counts"] < min_total_counts) |
        (stats["mean_nuclear_fraction"] < min_nuclear_fraction)
    ].index


# ---------------------------------------------------------------------------
# Console summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    cluster_summary: pd.DataFrame,
    low_quality_clusters: pd.Index,
    n_hard_removed: int,
    n_cell: int,
    n_empty: int,
    n_total: int,
    hard_filter_threshold: float,
    min_total_counts: float,
    min_nuclear_fraction: float,
    n_components: int,
    auto_n_components: bool,
) -> None:
    sep = "─" * 72

    print(f"\n{sep}")
    print("  dropletqcpy — identify_empty_droplets  |  GMM 2D Cluster Report")
    print(sep)

    # Key parameters box
    auto_tag = " (auto-detected)" if auto_n_components else " (manual)"
    print(textwrap.dedent(f"""
  Key parameters (adjust to tune classification):
  ┌──────────────────────────────────────────────────────────┐
  │  n_components         = {n_components:<4}{auto_tag:<25}       │
  │  hard_filter_threshold= {hard_filter_threshold:<6}  (log10 units)               │
  │  min_total_counts     = {min_total_counts:<8.0f}  (raw UMI count)            │
  │  min_nuclear_fraction = {min_nuclear_fraction:<6}                             │
  └──────────────────────────────────────────────────────────┘
    """).rstrip())

    # Per-cluster table with quality flag
    display = cluster_summary.copy()
    # "quality" column already exists from Step 5; just remap for display
    display["quality"] = display.index.map(
        lambda k: "✗  LOW" if k in low_quality_clusters else "✓  OK "
    )
    # Round floats for readability
    float_cols = display.select_dtypes(include=float).columns
    display[float_cols] = display[float_cols].round(4)

    print("\n  Per-cluster statistics:\n")
    # Add column header alignment manually for clean terminal output
    print(display.to_string(
        col_space=8,
        justify="right",
    ))

    # Removal summary
    pct_removed = 100.0 * n_empty / n_total if n_total > 0 else 0.0
    print(f"""
  ─ Removal summary ─────────────────────────────────────
  Hard-filtered (log10_counts ≤ {hard_filter_threshold})  : {n_hard_removed:>8,} bins
  Low-quality clusters (empty_droplet) : {n_empty:>8,} bins  ({pct_removed:.1f}%)
  Retained (cell)                      : {n_cell:>8,} bins  ({100-pct_removed:.1f}%)
  Total input                          : {n_total:>8,} bins
  ────────────────────────────────────────────────────────

  To adjust thresholds, rerun with e.g.:
    identify_empty_droplets(adata,
        n_components={n_components},       # set None for auto-detect
        min_total_counts=1000,
        min_nuclear_fraction=0.15,
    )
{sep}
""")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def identify_empty_droplets(
    adata: anndata.AnnData,
    *,
    nf_key: str = "nuclear_fraction",
    total_counts_key: Optional[str] = "total_counts",
    obs_key: str = "droplet_label",
    cluster_key: str = "gmm_cluster",
    n_components: Optional[int] = None,
    hard_filter_threshold: float = 1.0,
    min_total_counts: float = 500.0,
    min_nuclear_fraction: float = 0.1,
    random_state: int = 42,
) -> anndata.AnnData:
    """
    Identify empty droplets and low-quality bins using a 2D GMM on
    log10(total_counts) and nuclear_fraction.

    If total_counts is not present in adata.obs, sc.pp.calculate_qc_metrics
    is called automatically before clustering.

    Parameters
    ----------
    adata:
        AnnData object. Must contain nuclear_fraction in obs.
    nf_key:
        Column in adata.obs with nuclear fraction values.
        Default: ``"nuclear_fraction"``.
    total_counts_key:
        Column in adata.obs with total UMI counts. If None or not found,
        sc.pp.calculate_qc_metrics is run automatically.
        Default: ``"total_counts"``.
    obs_key:
        Output QC flag column written to adata.obs.
        Values: ``"cell"`` or ``"empty_droplet"``.
        Default: ``"droplet_label"``.
    cluster_key:
        Column name for GMM cluster labels in adata.obs.
        Default: ``"gmm_cluster"``.
    n_components:
        Number of GMM components. If None (default), auto-detected from the
        number of peaks in the KDE of nuclear_fraction (n_peaks + 1,
        clamped to [2, 6]). Override with an integer to fix manually.
    hard_filter_threshold:
        log10 threshold; bins with log10(total_counts+1) <= this are excluded
        before clustering. Default: ``1.0`` (total_counts <= 10).
    min_total_counts:
        Clusters with mean_total_counts < this are flagged low-quality.
        Uses raw UMI count (not log-transformed). Default: ``500.0``.
    min_nuclear_fraction:
        Clusters with mean_nuclear_fraction < this are flagged low-quality.
        Default: ``0.1``.
    random_state:
        Random seed for GMM reproducibility. Default: ``42``.

    Returns
    -------
    adata : anndata.AnnData
        Input adata with added obs columns:
          - adata.obs[cluster_key]  : GMM cluster label (-1 = hard-filtered)
          - adata.obs[obs_key]      : "cell" or "empty_droplet"
          - adata.obs["total_counts"], "n_genes_by_counts", "pct_counts_mt"
            (added if not already present)
        Results stored in adata.uns["dropletqcpy"]["empty_droplets"]:
          - "adata_clean" key is NOT stored (create it with adata_clean below)
          - "cluster_summary": cluster statistics as a dict
          - provenance metadata

        To get adata_clean and cluster_summary after the call:

        >>> adata_clean   = adata[adata.obs["droplet_label"] == "cell"].copy()
        >>> cluster_summary = pd.DataFrame(
        ...     adata.uns["dropletqcpy"]["empty_droplets"]["cluster_summary"]
        ... )

    Examples
    --------
    >>> adata = dp.identify_empty_droplets(adata)
    >>> adata_clean = adata[adata.obs["droplet_label"] == "cell"].copy()
    >>> # Re-run with adjusted thresholds if needed:
    >>> adata = dp.identify_empty_droplets(
    ...     adata, min_total_counts=1000, min_nuclear_fraction=0.15
    ... )
    """
    if nf_key not in adata.obs.columns:
        raise KeyError(
            f"'{nf_key}' not found in adata.obs. "
            "Run compute_nuclear_fraction() first."
        )

    # Step 0 — QC metrics + feature construction
    total_counts, log10_tc = _ensure_qc_metrics(adata, total_counts_key)
    nf = adata.obs[nf_key].to_numpy(dtype=np.float64)

    # Step 1 — Hard filter (before clustering)
    pass_filter = _hard_filter(log10_tc, hard_filter_threshold)
    n_hard_removed = int((~pass_filter).sum())
    logger.info(
        "Hard filter (log10_total_counts > %.1f): %d / %d bins pass.",
        hard_filter_threshold, int(pass_filter.sum()), adata.n_obs,
    )

    # Initialise output arrays
    gmm_labels_full = np.full(adata.n_obs, -1, dtype=np.int32)
    qc_flag_full    = np.full(adata.n_obs, _LABEL_EMPTY, dtype=object)

    # Only bins passing hard filter with finite NF enter the GMM
    finite_mask = pass_filter & np.isfinite(nf) & np.isfinite(log10_tc)
    n_model = int(finite_mask.sum())
    logger.info("Bins entering GMM: %d", n_model)

    # Auto-detect n_components if not specified
    auto_n_components = n_components is None
    if auto_n_components:
        n_components = _detect_n_components(nf[finite_mask])

    if n_model < n_components:
        raise RuntimeError(
            f"Only {n_model} bins pass the hard filter, which is fewer than "
            f"n_components={n_components}. Cannot fit GMM. "
            "Lower hard_filter_threshold or reduce n_components."
        )

    # Step 2 — Standardize clustering matrix
    X_raw    = np.column_stack([log10_tc[finite_mask], nf[finite_mask]])
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Step 3 — Gaussian Mixture Clustering
    logger.info("Fitting GMM (n_components=%d, covariance_type='diag').", n_components)
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        random_state=random_state,
        n_init=5,
        max_iter=300,
    )
    gmm.fit(X_scaled)
    labels_model = gmm.predict(X_scaled).astype(np.int32)
    gmm_labels_full[finite_mask] = labels_model

    # Step 4 — Cluster-level statistics
    cluster_summary = _cluster_stats(
        log10_tc[finite_mask],
        nf[finite_mask],
        total_counts[finite_mask],
        labels_model,
        n_components,
    )

    # Step 5 — Flag low-quality clusters
    low_quality_clusters = _flag_low_quality_clusters(
        cluster_summary, min_total_counts, min_nuclear_fraction
    )
    cluster_summary["quality"] = cluster_summary.index.map(
        lambda k: "low_quality" if k in low_quality_clusters else "high_quality"
    )

    # Step 6 — Build QC flag (vectorised)
    lq_set = set(int(x) for x in low_quality_clusters)
    is_high_quality = np.array([lbl not in lq_set for lbl in labels_model], dtype=bool)
    qc_flag_full[np.where(finite_mask)[0][is_high_quality]] = _LABEL_CELL

    # Write to adata.obs
    adata.obs[cluster_key] = gmm_labels_full
    adata.obs[obs_key]     = qc_flag_full

    n_cell  = int((qc_flag_full == _LABEL_CELL).sum())
    n_empty = int((qc_flag_full == _LABEL_EMPTY).sum())

    # Console summary
    _print_summary(
        cluster_summary=cluster_summary,
        low_quality_clusters=low_quality_clusters,
        n_hard_removed=n_hard_removed,
        n_cell=n_cell,
        n_empty=n_empty,
        n_total=adata.n_obs,
        hard_filter_threshold=hard_filter_threshold,
        min_total_counts=min_total_counts,
        min_nuclear_fraction=min_nuclear_fraction,
        n_components=n_components,
        auto_n_components=auto_n_components,
    )

    # Provenance metadata
    _write_uns(
        adata,
        obs_key=obs_key,
        cluster_key=cluster_key,
        n_components=n_components,
        hard_filter_threshold=hard_filter_threshold,
        min_total_counts=min_total_counts,
        min_nuclear_fraction=min_nuclear_fraction,
        low_quality_clusters=[int(x) for x in low_quality_clusters],
        n_cell=n_cell,
        n_empty=n_empty,
    )

    # Store cluster_summary in uns for downstream access
    adata.uns["dropletqcpy"]["empty_droplets"]["cluster_summary"] = (
        cluster_summary.reset_index().to_dict(orient="list")
    )

    n_clean = int((adata.obs[obs_key] == _LABEL_CELL).sum())
    logger.info(
        "Done. %d / %d bins retained as cells. "
        "Access clean subset with: adata[adata.obs['%s'] == 'cell'].copy()",
        n_clean, adata.n_obs, obs_key,
    )

    return adata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_uns(
    adata: anndata.AnnData,
    *,
    obs_key: str,
    cluster_key: str,
    n_components: int,
    hard_filter_threshold: float,
    min_total_counts: float,
    min_nuclear_fraction: float,
    low_quality_clusters: list,
    n_cell: int,
    n_empty: int,
) -> None:
    meta = {
        "computed_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "obs_key": obs_key,
        "cluster_key": cluster_key,
        "method": "GMM_2D",
        "parameters": {
            "n_components": n_components,
            "covariance_type": "diag",
            "hard_filter_threshold": hard_filter_threshold,
            "min_total_counts": min_total_counts,
            "min_nuclear_fraction": min_nuclear_fraction,
        },
        "low_quality_clusters": low_quality_clusters,
        "n_cell": n_cell,
        "n_empty_droplet": n_empty,
    }
    if "dropletqcpy" not in adata.uns:
        adata.uns["dropletqcpy"] = {}
    adata.uns["dropletqcpy"]["empty_droplets"] = meta
