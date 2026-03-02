"""
dropletqcpy
===========

An independent Python reimplementation of the DropletQC methodology.

Original method published in:
    Muskovic & Kim (2021). DropletQC: improved identification of empty
    droplets and damaged cells in single-cell RNA sequencing data.
    Genome Biology, 22, 329. https://doi.org/10.1186/s13059-021-02547-0

This package is not affiliated with or endorsed by the original authors.
"""

from dropletqcpy.nuclear_fraction import (
    compute_nuclear_fraction,
    compute_nuclear_fraction_from_bam,
)
from dropletqcpy.empty_droplets import identify_empty_droplets
from dropletqcpy.damaged_cells import identify_damaged_cells

__version__ = "0.1.0.dev0"

__all__ = [
    "compute_nuclear_fraction",
    "compute_nuclear_fraction_from_bam",
    "identify_empty_droplets",
    "identify_damaged_cells",
]
