"""
dropletqcpy.damaged_cells
=========================

Identify damaged cells using a per-cell-type Gaussian Mixture Model (GMM)
on nuclear fraction and log10(UMI), as described in:

    Muskovic & Kim (2021). DropletQC: improved identification of empty
    droplets and damaged cells in single-cell RNA sequencing data.
    Genome Biology, 22, 329. https://doi.org/10.1186/s13059-021-02547-0

This module is an independent Python reimplementation of the DropletQC
methodology. It is not affiliated with, endorsed by, or derived from
the original R package source code.

Model assumptions
-----------------
Features:
    X = [nf, log10(umi)]   — 2D, one row per cell

GMM covariance type:
    "tied" — all components share a single covariance matrix.
    This is the closest sklearn equivalent to "equal/shared variance across
    components" as described in the original method. It prevents a degenerate
    component from collapsing to zero variance on a single axis.

BIC-based model selection:
    Models with n_components ∈ {1, 2} are fitted; the model with the lower
    BIC is selected.

Three sequential credibility checks (applied per cell type):
    Check 1 — 2-component model must be selected by BIC.
    Check 2 — The high-NF component must also have lower mean log10(UMI)
               than the low-NF component (inverse NF–UMI relationship).
    Check 3 — Both NF and UMI separations must exceed configurable thresholds.

Only barcodes already labelled "cell" (by identify_empty_droplets) are
modelled. Empty droplet labels are never modified.
"""

from __future__ import annotations

import datetime
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import anndata
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

# Label constants
_LABEL_CELL = "cell"
_LABEL_EMPTY = "empty_droplet"
_LABEL_DAMAGED = "damaged_cell"

_MIN_CELLS_FOR_MODELING = 50
_MIN_COMPONENT_WEIGHT = 0.05  # components smaller than this are considered unreliable


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _CellTypeResult:
    """Per-cell-type modeling outcome."""
    cell_type: str
    n_cells: int
    n_components_selected: int
    checks_passed: bool
    damaged_indices: np.ndarray          # indices into the cell-type subset
    nf_cutoff: Optional[float] = None
    fail_reason: Optional[str] = None
    gmm: Optional[GaussianMixture] = None
    high_nf_component: Optional[int] = None  # component index (0 or 1)
    mean_nf: Optional[np.ndarray] = None
    mean_log10_umi: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# GMM fitting and model selection
# ---------------------------------------------------------------------------

def _fit_gmm(
    X: np.ndarray,
    n_components: int,
    n_init: int = 5,
    random_state: int = 0,
) -> Optional[GaussianMixture]:
    """
    Fit a tied-covariance GMM with ``n_components`` on feature matrix ``X``.

    Returns ``None`` if the EM algorithm fails to converge or raises an
    unexpected numerical error.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="tied",
        n_init=n_init,
        random_state=random_state,
        max_iter=200,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        try:
            gmm.fit(X)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GMM fitting failed: %s", exc)
            return None

    if not gmm.converged_:
        logger.warning("GMM did not converge (n_components=%d).", n_components)
        return None

    return gmm


def _select_model(
    X: np.ndarray,
    n_init: int = 5,
    random_state: int = 0,
) -> Tuple[Optional[GaussianMixture], int]:
    """
    Fit GMMs with 1 and 2 components; return the one with lower BIC.

    Returns
    -------
    (best_gmm, n_components_selected)
        ``best_gmm`` is ``None`` if both fits failed.
    """
    gmm1 = _fit_gmm(X, 1, n_init=n_init, random_state=random_state)
    gmm2 = _fit_gmm(X, 2, n_init=n_init, random_state=random_state)

    if gmm1 is None and gmm2 is None:
        return None, 1

    if gmm1 is None:
        return gmm2, 2

    if gmm2 is None:
        return gmm1, 1

    bic1 = gmm1.bic(X)
    bic2 = gmm2.bic(X)
    logger.debug("BIC: 1-component=%.2f, 2-component=%.2f", bic1, bic2)

    if bic2 < bic1:
        return gmm2, 2
    return gmm1, 1


# ---------------------------------------------------------------------------
# Credibility checks
# ---------------------------------------------------------------------------

def _run_checks(
    gmm: GaussianMixture,
    nf_sep: float,
    umi_sep_perc: float,
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Run the three sequential credibility checks on a 2-component GMM.

    Parameters
    ----------
    gmm:
        Fitted 2-component GaussianMixture.
    nf_sep:
        Minimum required difference in mean NF between the two components.
    umi_sep_perc:
        Minimum required percent decrease in mean UMI (original scale) for
        the high-NF (damaged) component relative to the low-NF (cell) component.

    Returns
    -------
    (passed, high_nf_idx, fail_reason)
        ``passed`` — True if all checks pass.
        ``high_nf_idx`` — index (0 or 1) of the damaged-cell component.
        ``fail_reason`` — human-readable explanation when ``passed`` is False.
    """
    means = gmm.means_          # shape (2, 2): rows=components, cols=[nf, log10_umi]
    weights = gmm.weights_      # shape (2,)

    mean_nf = means[:, 0]
    mean_log10_umi = means[:, 1]

    # Check 1 is enforced upstream (n_components == 2); nothing to do here.

    # Check 2: high-NF component must have lower mean log10(UMI)
    high_nf_idx = int(np.argmax(mean_nf))
    low_nf_idx = 1 - high_nf_idx

    if mean_log10_umi[high_nf_idx] >= mean_log10_umi[low_nf_idx]:
        return (
            False,
            None,
            "Check 2 failed: high-NF component does not have lower mean UMI.",
        )

    # Check 3a: NF separation
    nf_diff = mean_nf[high_nf_idx] - mean_nf[low_nf_idx]
    if nf_diff <= nf_sep:
        return (
            False,
            None,
            f"Check 3a failed: NF separation {nf_diff:.4f} <= threshold {nf_sep}.",
        )

    # Check 3b: UMI separation in original scale
    mean_umi_high = 10 ** mean_log10_umi[high_nf_idx]
    mean_umi_low = 10 ** mean_log10_umi[low_nf_idx]
    umi_pct_drop = (1.0 - mean_umi_high / mean_umi_low) * 100.0

    if umi_pct_drop <= umi_sep_perc:
        return (
            False,
            None,
            (
                f"Check 3b failed: UMI percent drop {umi_pct_drop:.1f}% "
                f"<= threshold {umi_sep_perc}%."
            ),
        )

    return True, high_nf_idx, None


# ---------------------------------------------------------------------------
# Per-cell-type modeling
# ---------------------------------------------------------------------------

def _model_cell_type(
    nf: np.ndarray,
    log10_umi: np.ndarray,
    cell_type: str,
    nf_sep: float,
    umi_sep_perc: float,
    min_component_weight: float,
    n_init: int,
    random_state: int,
) -> _CellTypeResult:
    """
    Run the full GMM pipeline for a single cell type.

    Parameters
    ----------
    nf, log10_umi:
        Feature arrays for cells of this type (already filtered: only "cell"
        labels, finite values). Length = n_cells_in_type.
    cell_type:
        Label used for logging and result tracking.

    Returns
    -------
    _CellTypeResult
        Contains per-cell damaged indices and diagnostics.
    """
    n_cells = len(nf)
    empty_result = _CellTypeResult(
        cell_type=cell_type,
        n_cells=n_cells,
        n_components_selected=1,
        checks_passed=False,
        damaged_indices=np.array([], dtype=np.intp),
    )

    if n_cells < _MIN_CELLS_FOR_MODELING:
        empty_result.fail_reason = f"Too few cells ({n_cells} < {_MIN_CELLS_FOR_MODELING})."
        logger.info("[%s] Skipped: %s", cell_type, empty_result.fail_reason)
        return empty_result

    X = np.column_stack([nf, log10_umi])   # shape (n_cells, 2)

    # Model selection via BIC
    best_gmm, n_selected = _select_model(X, n_init=n_init, random_state=random_state)

    if best_gmm is None:
        empty_result.fail_reason = "GMM fitting failed for both 1- and 2-component models."
        logger.warning("[%s] %s", cell_type, empty_result.fail_reason)
        return empty_result

    empty_result.n_components_selected = n_selected
    empty_result.gmm = best_gmm

    # Check 1: must select 2 components
    if n_selected != 2:
        empty_result.fail_reason = "BIC selected 1-component model."
        logger.info("[%s] %s All labelled 'cell'.", cell_type, empty_result.fail_reason)
        return empty_result

    # Minimum component weight guard
    weights = best_gmm.weights_
    if np.any(weights < min_component_weight):
        small_w = weights[weights < min_component_weight]
        empty_result.fail_reason = (
            f"Component weight(s) {small_w} below minimum {min_component_weight}."
        )
        logger.info("[%s] %s All labelled 'cell'.", cell_type, empty_result.fail_reason)
        return empty_result

    # Checks 2 & 3
    passed, high_nf_idx, fail_reason = _run_checks(best_gmm, nf_sep, umi_sep_perc)

    if not passed:
        empty_result.fail_reason = fail_reason
        logger.info("[%s] %s All labelled 'cell'.", cell_type, fail_reason)
        return empty_result

    # All checks passed — assign labels by component membership
    assignments = best_gmm.predict(X)   # shape (n_cells,)
    damaged_indices = np.where(assignments == high_nf_idx)[0]

    logger.info(
        "[%s] %d damaged cells identified (%.1f%% of type).",
        cell_type,
        len(damaged_indices),
        100.0 * len(damaged_indices) / n_cells,
    )

    return _CellTypeResult(
        cell_type=cell_type,
        n_cells=n_cells,
        n_components_selected=2,
        checks_passed=True,
        damaged_indices=damaged_indices,
        gmm=best_gmm,
        high_nf_component=high_nf_idx,
        mean_nf=best_gmm.means_[:, 0],
        mean_log10_umi=best_gmm.means_[:, 1],
        weights=weights,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def identify_damaged_cells(
    adata: anndata.AnnData,
    *,
    nf_key: str = "nuclear_fraction",
    umi_key: Optional[str] = None,
    label_key: str = "droplet_label",
    cell_type_key: str,
    obs_key: str = "cell_status",
    nf_sep: float = 0.15,
    umi_sep_perc: float = 50.0,
    min_component_weight: float = _MIN_COMPONENT_WEIGHT,
    n_init: int = 5,
    random_state: int = 0,
) -> anndata.AnnData:
    """
    Identify damaged cells within each cell type using a per-type GMM and
    write per-barcode labels into ``adata.obs``.

    Only barcodes previously labelled ``"cell"`` (in ``adata.obs[label_key]``)
    are modelled. Empty droplet labels are propagated unchanged.

    Parameters
    ----------
    adata:
        AnnData object. Must contain:

        - ``adata.obs[nf_key]`` — nuclear fraction (from
          ``dropletqcpy.compute_nuclear_fraction``).
        - ``adata.obs[label_key]`` — droplet labels (from
          ``dropletqcpy.identify_empty_droplets``).
        - ``adata.obs[cell_type_key]`` — cell type annotations (e.g. from
          clustering or reference-based annotation).

    nf_key:
        Column in ``adata.obs`` with nuclear fraction values.
        Default: ``"nuclear_fraction"``.
    umi_key:
        Column in ``adata.obs`` with total UMI counts. If ``None``,
        computed from ``adata.X.sum(axis=1)``. Default: ``None``.
    label_key:
        Column in ``adata.obs`` containing ``"cell"`` / ``"empty_droplet"``
        labels produced by ``identify_empty_droplets``.
        Default: ``"droplet_label"``.
    cell_type_key:
        Column in ``adata.obs`` containing cell type annotations.
        **Required** — no default, must be provided explicitly.
    obs_key:
        Output column written to ``adata.obs``. Values:
        ``"cell"`` / ``"empty_droplet"`` / ``"damaged_cell"``.
        Default: ``"cell_status"``.
    nf_sep:
        Minimum mean NF difference between the two GMM components required
        for a damaged-cell call (Check 3a). Default: ``0.15``.
    umi_sep_perc:
        Minimum percent decrease in mean UMI (original scale) of the
        high-NF component relative to the low-NF component (Check 3b).
        Default: ``50.0``.
    min_component_weight:
        Components with weight below this value are considered unreliable
        and the 2-component result is discarded. Default: ``0.05``.
    n_init:
        Number of GMM initialisations (passed to
        ``sklearn.mixture.GaussianMixture``). Default: ``5``.
    random_state:
        Random seed for reproducibility. Default: ``0``.

    Returns
    -------
    anndata.AnnData
        The input ``adata`` with:

        - ``adata.obs[obs_key]``: ``"cell"``, ``"empty_droplet"``, or
          ``"damaged_cell"`` per barcode.
        - ``adata.uns["dropletqcpy"]["damaged_cells"]``: provenance metadata
          and per-cell-type model summaries.

    Raises
    ------
    KeyError
        If any required ``adata.obs`` column is missing.

    Examples
    --------
    >>> adata = compute_nuclear_fraction(adata, "sample.loom")
    >>> adata = identify_empty_droplets(adata)
    >>> adata = identify_damaged_cells(adata, cell_type_key="leiden")
    >>> adata.obs["cell_status"].value_counts()
    """
    # --- Validate required columns ---
    for key in (nf_key, label_key, cell_type_key):
        if key not in adata.obs.columns:
            raise KeyError(
                f"'{key}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

    # --- Extract arrays ---
    nf_all: np.ndarray = adata.obs[nf_key].to_numpy(dtype=np.float64)
    labels_all: np.ndarray = adata.obs[label_key].to_numpy(dtype=object)
    ct_all: np.ndarray = adata.obs[cell_type_key].to_numpy(dtype=object)

    if umi_key is not None:
        if umi_key not in adata.obs.columns:
            raise KeyError(f"'{umi_key}' not found in adata.obs.")
        umi_all: np.ndarray = adata.obs[umi_key].to_numpy(dtype=np.float64)
    else:
        logger.info("umi_key not specified; computing UMI from adata.X.sum(axis=1).")
        umi_all = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float64)

    # --- Initialise output from existing droplet labels ---
    # "empty_droplet" barcodes are copied as-is; "cell" barcodes start as "cell"
    # and may be updated to "damaged_cell" below.
    output_labels = np.where(
        labels_all == _LABEL_EMPTY,
        _LABEL_EMPTY,
        _LABEL_CELL,
    ).astype(object)

    # --- Identify the "cell" subset for modeling ---
    cell_mask = labels_all == _LABEL_CELL
    cell_global_idx = np.where(cell_mask)[0]

    nf_cells = nf_all[cell_mask]
    umi_cells = umi_all[cell_mask]
    ct_cells = ct_all[cell_mask]

    # Guard: log10(umi) is undefined for umi <= 0
    valid_umi = umi_cells > 0
    log10_umi_cells = np.full(len(umi_cells), np.nan, dtype=np.float64)
    log10_umi_cells[valid_umi] = np.log10(umi_cells[valid_umi])

    # --- Per-cell-type modeling ---
    cell_types = np.unique(ct_cells)
    results: Dict[str, _CellTypeResult] = {}

    for ct in cell_types:
        ct_mask = ct_cells == ct

        # Within-type valid mask: finite nf and finite log10_umi
        ct_nf = nf_cells[ct_mask]
        ct_log10_umi = log10_umi_cells[ct_mask]
        finite_mask = np.isfinite(ct_nf) & np.isfinite(ct_log10_umi)

        # Indices within the ct subset, then mapped back to cell_global_idx
        ct_local_idx = np.where(ct_mask)[0]        # position within cell subset
        finite_local_idx = ct_local_idx[finite_mask]

        result = _model_cell_type(
            nf=ct_nf[finite_mask],
            log10_umi=ct_log10_umi[finite_mask],
            cell_type=ct,
            nf_sep=nf_sep,
            umi_sep_perc=umi_sep_perc,
            min_component_weight=min_component_weight,
            n_init=n_init,
            random_state=random_state,
        )
        results[ct] = result

        if result.checks_passed and len(result.damaged_indices) > 0:
            # damaged_indices are positions within the finite subset of this ct
            damaged_in_finite = result.damaged_indices
            damaged_cell_subset_idx = finite_local_idx[damaged_in_finite]
            damaged_global_idx = cell_global_idx[damaged_cell_subset_idx]
            output_labels[damaged_global_idx] = _LABEL_DAMAGED

    # --- Write to adata ---
    adata.obs[obs_key] = output_labels

    n_cell = int((output_labels == _LABEL_CELL).sum())
    n_damaged = int((output_labels == _LABEL_DAMAGED).sum())
    n_empty = int((output_labels == _LABEL_EMPTY).sum())
    logger.info(
        "Final labels — cell: %d, damaged_cell: %d, empty_droplet: %d.",
        n_cell, n_damaged, n_empty,
    )

    _write_uns(
        adata,
        obs_key=obs_key,
        nf_sep=nf_sep,
        umi_sep_perc=umi_sep_perc,
        min_component_weight=min_component_weight,
        n_cell=n_cell,
        n_damaged=n_damaged,
        n_empty=n_empty,
        results=results,
    )

    return adata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_uns(
    adata: anndata.AnnData,
    *,
    obs_key: str,
    nf_sep: float,
    umi_sep_perc: float,
    min_component_weight: float,
    n_cell: int,
    n_damaged: int,
    n_empty: int,
    results: Dict[str, _CellTypeResult],
) -> None:
    """Write provenance metadata to adata.uns['dropletqcpy']['damaged_cells']."""

    per_type_summary = {}
    for ct, r in results.items():
        entry: dict = {
            "n_cells": r.n_cells,
            "n_components_selected": r.n_components_selected,
            "checks_passed": r.checks_passed,
            "n_damaged": int(len(r.damaged_indices)) if r.checks_passed else 0,
        }
        if r.fail_reason:
            entry["fail_reason"] = r.fail_reason
        if r.checks_passed and r.mean_nf is not None:
            entry["mean_nf_per_component"] = r.mean_nf.tolist()
            entry["mean_log10_umi_per_component"] = r.mean_log10_umi.tolist()
            entry["component_weights"] = r.weights.tolist()
            entry["damaged_component_index"] = r.high_nf_component
        per_type_summary[str(ct)] = entry

    meta = {
        "computed_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "obs_key": obs_key,
        "parameters": {
            "nf_sep": nf_sep,
            "umi_sep_perc": umi_sep_perc,
            "min_component_weight": min_component_weight,
            "covariance_type": "tied",
        },
        "n_cell": n_cell,
        "n_damaged_cell": n_damaged,
        "n_empty_droplet": n_empty,
        "per_cell_type": per_type_summary,
    }

    if "dropletqcpy" not in adata.uns:
        adata.uns["dropletqcpy"] = {}
    adata.uns["dropletqcpy"]["damaged_cells"] = meta


# ---------------------------------------------------------------------------
# Minimal example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import scanpy as sc
    from dropletqcpy.nuclear_fraction import compute_nuclear_fraction
    from dropletqcpy.empty_droplets import identify_empty_droplets

    adata = sc.read_h5ad("sample.h5ad")

    # Step 1: nuclear fraction from loom
    adata = compute_nuclear_fraction(adata, "sample.loom")

    # Step 2: empty droplet classification
    adata = identify_empty_droplets(adata)

    # Step 3: damaged cell detection — cell_type_key is required
    # (e.g. "leiden" from sc.tl.leiden, or "cell_type" from reference annotation)
    adata = identify_damaged_cells(adata, cell_type_key="leiden")

    print(adata.obs["cell_status"].value_counts())
    print(adata.uns["dropletqcpy"]["damaged_cells"]["per_cell_type"])
