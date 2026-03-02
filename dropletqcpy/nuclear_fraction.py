"""
dropletqcpy.nuclear_fraction
============================

Compute the nuclear fraction (NF) per cell barcode as defined in:

    Muskovic & Kim (2021). DropletQC: improved identification of empty
    droplets and damaged cells in single-cell RNA sequencing data.
    Genome Biology, 22, 329. https://doi.org/10.1186/s13059-021-02547-0

This module is an independent Python reimplementation of the DropletQC
methodology. It is not affiliated with, endorsed by, or derived from
the original R package source code.

Nuclear fraction is defined as:
    NF = unspliced / (spliced + unspliced)

Two input paths are supported:
  1. Loom file (velocyto output) — direct spliced/unspliced matrices.
  2. BAM file + GTF — streaming computation (see bam_counter.py).
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Callable, Optional, Union

import anndata
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path 1: Loom-based nuclear fraction
# ---------------------------------------------------------------------------

def compute_nuclear_fraction(
    adata: anndata.AnnData,
    loom_path: Union[str, Path],
    *,
    spliced_layer: str = "spliced",
    unspliced_layer: str = "unspliced",
    barcode_key: str = "CellID",
    barcode_transform: Optional[Callable[[str], str]] = None,
    adata_barcode_transform: Optional[Callable[[str], str]] = None,
    obs_key: str = "nuclear_fraction",
    min_counts: int = 1,
) -> anndata.AnnData:
    """
    Compute per-cell nuclear fraction from a velocyto loom file and write
    the result into adata.obs.

    Both loom and adata barcodes can be independently transformed before
    matching, handling the common case where velocyto and CellRanger produce
    different barcode formats.

    A typical real-world mismatch (velocyto + CellRanger):
      loom:  "OES272835:AAAGGGCCAATCACGTx"
      adata: "AAAGGGCCAATCACGT-1"

    Resolved with:
      barcode_transform=lambda bc: bc.split(":")[-1].rstrip("x"),
      adata_barcode_transform=lambda bc: bc.rsplit("-", 1)[0],

    Note: adata.obs_names are never modified. adata_barcode_transform is
    applied only internally for the lookup key.

    Parameters
    ----------
    adata : AnnData
    loom_path : str or Path
    spliced_layer : str, default "spliced"
    unspliced_layer : str, default "unspliced"
    barcode_key : str, default "CellID"
    barcode_transform : callable, optional
        Applied to each loom barcode before matching.
    adata_barcode_transform : callable, optional
        Applied to each adata barcode before matching (obs_names unchanged).
    obs_key : str, default "nuclear_fraction"
    min_counts : int, default 1

    Returns
    -------
    AnnData
        adata.obs[obs_key] : nuclear fraction per cell (NaN if unmatched)
        adata.uns["dropletqcpy"]["nuclear_fraction"] : provenance metadata
    """
    try:
        import loompy
    except ImportError as exc:
        raise ImportError(
            "loompy is required for loom-based computation. "
            "Install it with: pip install loompy"
        ) from exc

    loom_path = Path(loom_path)
    logger.info("Reading loom file: %s", loom_path)

    with loompy.connect(str(loom_path), mode="r") as ds:
        for layer_name in (spliced_layer, unspliced_layer):
            if layer_name not in ds.layers:
                raise KeyError(
                    f"Layer '{layer_name}' not found in loom file. "
                    f"Available layers: {list(ds.layers.keys())}"
                )

        if barcode_key not in ds.ca:
            raise KeyError(
                f"Column attribute '{barcode_key}' not found. "
                f"Available attributes: {list(ds.ca.keys())}"
            )

        loom_barcodes: np.ndarray = ds.ca[barcode_key]
        logger.info("Loom file: %d cells, %d genes", ds.shape[1], ds.shape[0])

        # Transform loom barcodes before building the lookup index
        if barcode_transform is not None:
            loom_barcodes = np.array([barcode_transform(bc) for bc in loom_barcodes])

        loom_index: dict[str, int] = {bc: i for i, bc in enumerate(loom_barcodes)}

        # Optionally transform adata barcodes for lookup only (obs_names unchanged)
        raw_adata_barcodes = adata.obs_names.tolist()
        if adata_barcode_transform is not None:
            lookup_keys = [adata_barcode_transform(bc) for bc in raw_adata_barcodes]
        else:
            lookup_keys = raw_adata_barcodes

        col_indices = np.array(
            [loom_index.get(bc, -1) for bc in lookup_keys], dtype=np.int64
        )
        matched_mask = col_indices >= 0
        matched_cols = col_indices[matched_mask]

        n_matched = int(matched_mask.sum())
        n_total = adata.n_obs
        logger.info("%d / %d adata barcodes matched in loom file.", n_matched, n_total)

        if n_matched == 0:
            logger.warning(
                "No barcodes matched. Check barcode format or use "
                "barcode_transform / adata_barcode_transform."
            )
            adata.obs[obs_key] = np.nan
            _write_uns(adata, obs_key, n_total, n_total)
            return adata

        # h5py requires column indices to be strictly monotonically increasing.
        # Sort matched_cols, load in sorted order, then restore original adata order.
        sort_order = np.argsort(matched_cols)       # positions that sort matched_cols
        sorted_cols = matched_cols[sort_order]      # strictly increasing loom col idx
        invert_order = np.argsort(sort_order)       # restores original adata order

        spliced_sorted = ds.layers[spliced_layer][:, sorted_cols].sum(axis=0)
        unspliced_sorted = ds.layers[unspliced_layer][:, sorted_cols].sum(axis=0)

        spliced_matched = spliced_sorted[invert_order]
        unspliced_matched = unspliced_sorted[invert_order]

    spliced_matched = spliced_matched.astype(np.float64)
    unspliced_matched = unspliced_matched.astype(np.float64)
    total_matched = spliced_matched + unspliced_matched

    nf_matched = np.where(
        total_matched >= min_counts,
        unspliced_matched / total_matched,
        np.nan,
    )

    nf_full = np.full(n_total, np.nan, dtype=np.float64)
    nf_full[matched_mask] = nf_matched

    adata.obs[obs_key] = nf_full

    n_missing = int(np.isnan(nf_full).sum())
    if n_missing > 0:
        logger.warning(
            "%d / %d cells have NaN nuclear_fraction "
            "(unmatched barcodes or total_counts < %d).",
            n_missing, n_total, min_counts,
        )

    logger.info(
        "Nuclear fraction written to adata.obs['%s']. Mean=%.4f, Median=%.4f.",
        obs_key,
        float(np.nanmean(nf_full)),
        float(np.nanmedian(nf_full)),
    )

    _write_uns(adata, obs_key, n_missing, n_total)
    return adata


# ---------------------------------------------------------------------------
# Path 2: BAM-based nuclear fraction
# ---------------------------------------------------------------------------

def compute_nuclear_fraction_from_bam(
    adata: anndata.AnnData,
    bam_path: Union[str, Path],
    *,
    gtf_path: Optional[Union[str, Path]] = None,
    cb_tag: str = "CB",
    re_tag: str = "RE",
    exon_tag: str = "E",
    intron_tag: str = "N",
    barcode_transform: Optional[Callable[[str], str]] = None,
    obs_key: str = "nuclear_fraction",
    min_counts: int = 1,
) -> anndata.AnnData:
    """
    Compute per-cell nuclear fraction by streaming a BAM file and write
    the result into adata.obs.

    The classification strategy is selected automatically:

    - CellRanger BAM (RE tag present): reads are classified directly from
      the RE tag (E=exonic, N=intronic). No GTF required.
    - Generic BAM (no RE tag): reads are classified by overlap with
      exon/intron intervals from the provided GTF file.

    NF = intronic reads / (intronic reads + exonic reads)

    Parameters
    ----------
    adata:
        AnnData object. Barcodes are read from adata.obs_names.
    bam_path:
        Path to a coordinate-sorted, indexed BAM file (.bai must be present).
    gtf_path:
        Path to a GTF annotation file (Ensembl/GENCODE). Required only when
        the BAM does not contain the RE tag. Default: None.
    cb_tag:
        BAM tag for the cell barcode. Default: "CB".
    re_tag:
        BAM tag for region type (CellRanger). Default: "RE".
    exon_tag:
        RE tag value marking exonic reads. Default: "E".
    intron_tag:
        RE tag value marking intronic reads. Default: "N".
    barcode_transform:
        Optional callable applied to BAM barcodes before matching against
        adata.obs_names. Default: None.
    obs_key:
        Column name written to adata.obs. Default: "nuclear_fraction".
    min_counts:
        Cells with (exonic + intronic) < min_counts receive NaN. Default: 1.

    Returns
    -------
    AnnData
        adata.obs[obs_key]: nuclear fraction per cell.
        adata.uns["dropletqcpy"]["nuclear_fraction"]: provenance metadata.

    Raises
    ------
    ImportError
        If pysam is not installed.
    ValueError
        If the BAM has no RE tag and gtf_path is not provided.
    """
    from dropletqcpy.bam_counter import count_spliced_unspliced

    barcodes = adata.obs_names.tolist()
    n_total = adata.n_obs

    logger.info(
        "Starting BAM-based nuclear fraction computation for %d barcodes.", n_total
    )

    exonic, intronic, nuclear_fraction = count_spliced_unspliced(
        bam_path=bam_path,
        barcodes=barcodes,
        gtf_path=gtf_path,
        cb_tag=cb_tag,
        re_tag=re_tag,
        exon_tag=exon_tag,
        intron_tag=intron_tag,
        barcode_transform=barcode_transform,
        min_counts=min_counts,
    )

    adata.obs[obs_key] = nuclear_fraction

    n_missing = int(np.isnan(nuclear_fraction).sum())
    if n_missing > 0:
        logger.warning(
            "%d / %d cells have NaN nuclear_fraction "
            "(no reads counted or total < %d).",
            n_missing, n_total, min_counts,
        )

    logger.info(
        "Nuclear fraction written to adata.obs['%s']. Mean=%.4f, Median=%.4f.",
        obs_key,
        float(np.nanmean(nuclear_fraction)),
        float(np.nanmedian(nuclear_fraction)),
    )

    _write_uns(adata, obs_key, n_missing, n_total)
    return adata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_uns(
    adata: anndata.AnnData,
    obs_key: str,
    n_missing: int,
    n_total: int,
) -> None:
    """Write provenance metadata to adata.uns['dropletqcpy']['nuclear_fraction']."""
    nf_values = adata.obs[obs_key].to_numpy().astype(np.float64)
    meta = {
        "computed_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "obs_key": obs_key,
        "n_cells": n_total,
        "n_missing": n_missing,
        "mean_nuclear_fraction": float(np.nanmean(nf_values)),
        "median_nuclear_fraction": float(np.nanmedian(nf_values)),
    }
    if "dropletqcpy" not in adata.uns:
        adata.uns["dropletqcpy"] = {}
    adata.uns["dropletqcpy"]["nuclear_fraction"] = meta
