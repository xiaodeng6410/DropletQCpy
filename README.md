## Background and Attribution

The DropletQC methodology was originally developed and published by:

Young MD, Behjati S.  
**DropletQC: improved identification of empty droplets and damaged cells in single-cell RNA-seq data.**  
*Genome Biology* (2021) 22:293.  
https://link.springer.com/article/10.1186/s13059-021-02547-0
        
        
        
        
        
        

The official implementation of DropletQC is provided as an R package:
https://github.com/powellgenomicslab/DropletQC_paper

The original software is designed for the R ecosystem and integrates with Bioconductor-based workflows. While this provides strong compatibility within R-based pipelines, it limits direct integration with Python-based single-cell analysis frameworks such as Scanpy and AnnData, which are widely used in the Python scientific computing ecosystem.

To address this limitation, we developed **dropletqcpy**, an independent Python reimplementation of the DropletQC methodology. This project:

- Reimplements the published algorithmic framework in Python
- Preserves the original statistical definitions (nuclear fraction, KDE-based trough detection, rescue logic)
- Integrates natively with AnnData objects
- Enables seamless use within Scanpy-based workflows

This is a clean-room reimplementation based solely on the published methodology and documented behavior of DropletQC. No source code from the original R implementation has been reused.

If you use `dropletqcpy`, please cite the original DropletQC publication.
