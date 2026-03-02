"""
dropletqcpy.bam_counter
=======================

Stream a BAM file to compute per-cell exonic / intronic read counts,
then derive the nuclear fraction (NF) per barcode.

    NF = intronic reads / (intronic reads + exonic reads)

Two classification strategies are supported, selected automatically:

Strategy 1 — RE tag (CellRanger BAM, no GTF needed)
----------------------------------------------------
CellRanger adds a region-type tag (default: "RE") to every aligned read:
    E  →  exonic
    N  →  intronic
    I  →  intergenic (excluded from NF denominator)

If the first mapped read in the BAM carries the RE tag, this strategy
is used automatically.

Strategy 2 — GTF annotation (generic BAM, gtf_path required)
-------------------------------------------------------------
If the BAM does not contain an RE tag, each read is classified by
comparing its genomic coordinates against exon and intron intervals
extracted from the provided GTF file:
    read overlaps exon only   →  exonic
    read overlaps intron       →  intronic
    otherwise                  →  intergenic (excluded)

Pass ``gtf_path`` to force this strategy or to use it as a fallback.

Reference
---------
Muskovic & Kim (2021). DropletQC: improved identification of empty
droplets and damaged cells in single-cell RNA-seq data.
Genome Biology, 22, 329. https://doi.org/10.1186/s13059-021-02547-0
"""

from __future__ import annotations

import bisect
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GTF parsing (Strategy 2 only)
# ---------------------------------------------------------------------------

def _parse_attribute(attr_string: str, key: str) -> Optional[str]:
    match = re.search(rf'{key}\s+"([^"]+)"', attr_string)
    return match.group(1) if match else None


def _build_interval_index(
    gtf_path: Union[str, Path],
) -> Tuple[
    Dict[str, List[Tuple[int, int]]],
    Dict[str, List[Tuple[int, int]]],
]:
    """
    Parse a GTF file and return sorted exon and intron interval lists
    keyed by chromosome name.

    Introns are derived as gaps between consecutive exons within the
    same transcript. All coordinates are 0-based half-open [start, end).

    Parameters
    ----------
    gtf_path:
        Path to an Ensembl or GENCODE GTF file.

    Returns
    -------
    exons_by_chrom : dict
        chrom → sorted list of (start, end) exon intervals.
    introns_by_chrom : dict
        chrom → sorted list of (start, end) intron intervals.
    """
    gtf_path = Path(gtf_path)
    logger.info("Parsing GTF: %s", gtf_path)

    # transcript_id → [(chrom, start0, end0), ...]
    transcript_exons: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)

    with open(gtf_path, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue
            chrom = fields[0]
            start0 = int(fields[3]) - 1   # GTF is 1-based inclusive
            end0 = int(fields[4])          # convert to 0-based half-open
            tx_id = _parse_attribute(fields[8], "transcript_id")
            if tx_id is None:
                continue
            transcript_exons[tx_id].append((chrom, start0, end0))

    exon_set: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)
    intron_set: Dict[str, Set[Tuple[int, int]]] = defaultdict(set)

    for exons in transcript_exons.values():
        chrom = exons[0][0]
        intervals = sorted((s, e) for (_, s, e) in exons)
        for s, e in intervals:
            exon_set[chrom].add((s, e))
        for i in range(len(intervals) - 1):
            intron_start = intervals[i][1]
            intron_end = intervals[i + 1][0]
            if intron_end > intron_start:
                intron_set[chrom].add((intron_start, intron_end))

    exons_by_chrom = {c: sorted(v) for c, v in exon_set.items()}
    introns_by_chrom = {c: sorted(v) for c, v in intron_set.items()}

    logger.info(
        "GTF parsed: %d chromosomes, %d unique exons, %d unique introns.",
        len(exons_by_chrom),
        sum(len(v) for v in exons_by_chrom.values()),
        sum(len(v) for v in introns_by_chrom.values()),
    )
    return exons_by_chrom, introns_by_chrom


def _overlaps_any(
    read_start: int,
    read_end: int,
    intervals: List[Tuple[int, int]],
) -> bool:
    """Return True if [read_start, read_end) overlaps any interval."""
    idx = bisect.bisect_right(intervals, (read_end, read_end)) - 1
    while idx >= 0:
        iv_start, iv_end = intervals[idx]
        if iv_end <= read_start:
            break
        if iv_start < read_end:
            return True
        idx -= 1
    return False


def _fully_within_any(
    read_start: int,
    read_end: int,
    intervals: List[Tuple[int, int]],
) -> bool:
    """Return True if [read_start, read_end) is fully inside any interval."""
    idx = bisect.bisect_right(intervals, (read_start + 1, read_start + 1)) - 1
    while idx >= 0:
        iv_start, iv_end = intervals[idx]
        if iv_start > read_start:
            idx -= 1
            continue
        if iv_start <= read_start and iv_end >= read_end:
            return True
        break
    return False


# ---------------------------------------------------------------------------
# BAM streaming — Strategy 1: RE tag
# ---------------------------------------------------------------------------

def _stream_re_tag(
    bam,
    barcode_to_idx: Dict[str, int],
    exonic_counts: np.ndarray,
    intronic_counts: np.ndarray,
    cb_tag: str,
    re_tag: str,
    exon_tag: str,
    intron_tag: str,
    barcode_transform: Optional[Callable[[str], str]],
) -> Tuple[int, int, int, int, int]:
    """Stream BAM and classify reads via RE tag. Returns diagnostic counts."""
    n_reads = n_counted = n_no_cb = n_no_re = n_not_whitelisted = 0

    for read in bam.fetch(until_eof=True):
        n_reads += 1
        if (
            read.is_unmapped or read.is_secondary
            or read.is_supplementary or read.is_qcfail
        ):
            continue
        if not read.has_tag(cb_tag):
            n_no_cb += 1
            continue
        raw_bc: str = read.get_tag(cb_tag)
        bc = barcode_transform(raw_bc) if barcode_transform else raw_bc
        cell_idx = barcode_to_idx.get(bc, -1)
        if cell_idx < 0:
            n_not_whitelisted += 1
            continue
        if not read.has_tag(re_tag):
            n_no_re += 1
            continue
        region: str = read.get_tag(re_tag)
        if region == exon_tag:
            exonic_counts[cell_idx] += 1
            n_counted += 1
        elif region == intron_tag:
            intronic_counts[cell_idx] += 1
            n_counted += 1
        # intergenic → excluded from denominator

    return n_reads, n_counted, n_no_cb, n_no_re, n_not_whitelisted


# ---------------------------------------------------------------------------
# BAM streaming — Strategy 2: GTF annotation
# ---------------------------------------------------------------------------

def _stream_gtf_annotation(
    bam,
    barcode_to_idx: Dict[str, int],
    exonic_counts: np.ndarray,
    intronic_counts: np.ndarray,
    exons_by_chrom: Dict[str, List[Tuple[int, int]]],
    introns_by_chrom: Dict[str, List[Tuple[int, int]]],
    cb_tag: str,
    barcode_transform: Optional[Callable[[str], str]],
) -> Tuple[int, int, int, int]:
    """Stream BAM and classify reads via GTF interval overlap."""
    n_reads = n_counted = n_no_cb = n_not_whitelisted = 0

    for read in bam.fetch(until_eof=True):
        n_reads += 1
        if (
            read.is_unmapped or read.is_secondary
            or read.is_supplementary or read.is_qcfail
        ):
            continue
        if not read.has_tag(cb_tag):
            n_no_cb += 1
            continue
        raw_bc: str = read.get_tag(cb_tag)
        bc = barcode_transform(raw_bc) if barcode_transform else raw_bc
        cell_idx = barcode_to_idx.get(bc, -1)
        if cell_idx < 0:
            n_not_whitelisted += 1
            continue

        chrom = read.reference_name
        read_start = read.reference_start
        read_end = read.reference_end
        if read_end is None:
            continue

        exons = exons_by_chrom.get(chrom, [])
        introns = introns_by_chrom.get(chrom, [])

        if _fully_within_any(read_start, read_end, exons):
            exonic_counts[cell_idx] += 1
            n_counted += 1
        elif _overlaps_any(read_start, read_end, introns):
            intronic_counts[cell_idx] += 1
            n_counted += 1
        # else: intergenic → excluded

    return n_reads, n_counted, n_no_cb, n_not_whitelisted


# ---------------------------------------------------------------------------
# Strategy detection
# ---------------------------------------------------------------------------

def _has_re_tag(bam_path: str, re_tag: str, n_probe: int = 200) -> bool:
    """
    Check whether the BAM file contains the RE tag by sampling the first
    ``n_probe`` mapped reads. Returns True if any carry the tag.
    """
    try:
        import pysam
    except ImportError:
        return False

    seen = 0
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.has_tag(re_tag):
                return True
            seen += 1
            if seen >= n_probe:
                break
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def count_spliced_unspliced(
    bam_path: Union[str, Path],
    barcodes: List[str],
    *,
    cb_tag: str = "CB",
    re_tag: str = "RE",
    exon_tag: str = "E",
    intron_tag: str = "N",
    gtf_path: Optional[Union[str, Path]] = None,
    barcode_transform: Optional[Callable[[str], str]] = None,
    min_counts: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Count exonic and intronic reads per barcode and compute nuclear fraction.

    Strategy is selected automatically:

    - If the BAM file contains the RE tag (CellRanger output): reads are
      classified directly from the tag. ``gtf_path`` is not used.
    - If the RE tag is absent: ``gtf_path`` must be provided. Reads are
      classified by overlap with exon/intron intervals from the GTF.

    Parameters
    ----------
    bam_path:
        Path to a coordinate-sorted BAM file (.bai must be present).
    barcodes:
        Ordered list of valid cell barcodes (from ``adata.obs_names``).
    cb_tag:
        BAM tag for the cell barcode. Default: ``"CB"``.
    re_tag:
        BAM tag for region type (CellRanger). Default: ``"RE"``.
    exon_tag:
        RE tag value marking exonic reads. Default: ``"E"``.
    intron_tag:
        RE tag value marking intronic reads. Default: ``"N"``.
    gtf_path:
        Path to a GTF annotation file (Ensembl/GENCODE). Required when
        the BAM does not contain the RE tag. Default: ``None``.
    barcode_transform:
        Optional callable applied to raw BAM barcodes before lookup.
        Default: ``None``.
    min_counts:
        Barcodes with (exonic + intronic) < min_counts receive NaN NF.
        Default: ``1``.

    Returns
    -------
    exonic_counts : np.ndarray, shape (n_barcodes,)
    intronic_counts : np.ndarray, shape (n_barcodes,)
    nuclear_fraction : np.ndarray, shape (n_barcodes,)
        NaN where (exonic + intronic) < min_counts.

    Raises
    ------
    ImportError
        If pysam is not installed.
    ValueError
        If the BAM has no RE tag and ``gtf_path`` is not provided.
    """
    try:
        import pysam
    except ImportError as exc:
        raise ImportError(
            "pysam is required for BAM-based computation. "
            "Install it with: pip install pysam"
        ) from exc

    bam_path = str(bam_path)
    n = len(barcodes)
    barcode_to_idx: Dict[str, int] = {bc: i for i, bc in enumerate(barcodes)}

    exonic_counts = np.zeros(n, dtype=np.float64)
    intronic_counts = np.zeros(n, dtype=np.float64)

    # Auto-detect strategy
    use_re_tag = _has_re_tag(bam_path, re_tag)

    if use_re_tag:
        logger.info("Strategy: RE tag ('%s') detected — no GTF required.", re_tag)
        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
            n_reads, n_counted, n_no_cb, n_no_re, n_not_wl = _stream_re_tag(
                bam, barcode_to_idx, exonic_counts, intronic_counts,
                cb_tag, re_tag, exon_tag, intron_tag, barcode_transform,
            )
        logger.info(
            "BAM complete (RE tag): total=%d, counted=%d, "
            "no_cb=%d, no_re=%d, not_whitelisted=%d.",
            n_reads, n_counted, n_no_cb, n_no_re, n_not_wl,
        )

    else:
        logger.info("Strategy: RE tag not found — falling back to GTF annotation.")
        if gtf_path is None:
            raise ValueError(
                "The BAM file does not contain the RE tag and no gtf_path was "
                "provided. Please supply a GTF file via the gtf_path argument."
            )
        exons_by_chrom, introns_by_chrom = _build_interval_index(gtf_path)
        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
            n_reads, n_counted, n_no_cb, n_not_wl = _stream_gtf_annotation(
                bam, barcode_to_idx, exonic_counts, intronic_counts,
                exons_by_chrom, introns_by_chrom, cb_tag, barcode_transform,
            )
        logger.info(
            "BAM complete (GTF): total=%d, counted=%d, "
            "no_cb=%d, not_whitelisted=%d.",
            n_reads, n_counted, n_no_cb, n_not_wl,
        )

    total = exonic_counts + intronic_counts
    nuclear_fraction = np.where(
        total >= min_counts,
        intronic_counts / total,
        np.nan,
    )

    return exonic_counts, intronic_counts, nuclear_fraction
