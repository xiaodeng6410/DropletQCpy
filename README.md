## Background and Attribution

The DropletQC methodology was originally developed and published by:

Young MD, Behjati S.  
**DropletQC: improved identification of empty droplets and damaged cells in single-cell RNA-seq data.**  
*Genome Biology* (2021) 22:293.  
https://link.springer.com/article/10.1186/s13059-021-02547-0
        
        
        
        
        
        
        
        
        
        
        
        

The official implementation of DropletQC is provided as an R package:
[https://github.com/powellgenomicslab/DropletQC_paper](https://github.com/powellgenomicslab/DropletQC)

The original software is designed for the R ecosystem and integrates with Bioconductor-based workflows. While this provides strong compatibility within R-based pipelines, it limits direct integration with Python-based single-cell analysis frameworks such as Scanpy and AnnData, which are widely used in the Python scientific computing ecosystem.

To address this limitation, we developed **dropletqcpy**, an independent Python reimplementation of the DropletQC methodology. This project:

- Reimplements the published algorithmic framework in Python
- Preserves the original statistical definitions (nuclear fraction, KDE-based trough detection, rescue logic)
- Integrates natively with AnnData objects
- Enables seamless use within Scanpy-based workflows

This is a clean-room reimplementation based solely on the published methodology and documented behavior of DropletQC. No source code from the original R implementation has been reused.

If you use `dropletqcpy`, please cite the original DropletQC publication.

⚠️ Development Status & Known Limitations
🚧 Project Status: Work in Progress
This tool is currently in the early development stage (Alpha/Beta). While core functionalities are operational, several modules are still undergoing iteration and optimization. We welcome community feedback, issue reports, and Pull Requests to help improve the project.
🔍 Known Limitation: Broken Cell Detection
Currently, this tool has not yet identified an optimal solution for detecting and filtering "broken cells" (low-quality cells or cell debris).
The current algorithms for identifying low-quality cells or fragments may lack precision.
Recommendation: We advise users to combine this tool with other specialized methods (e.g., DoubletFinder, Scrublet, or manual thresholding based on mitochondrial percentage) to effectively filter out broken cells during quality control.
Improving this functionality is a top priority on our development roadmap.
✨ Key Advantage:
1.Empty Droplet Detection
Despite the limitation above, this tool excels in Empty Droplet Detection:
Superior Sensitivity: Compared to the R version of DropletQC, this Python implementation utilizes an improved statistical approach. It more accurately distinguishes the boundary between low-count real cells and empty droplets.
Streamlined Workflow: Eliminating the need to switch between R and Python environments, this tool offers a smoother, automated pipeline. It enables users to remove background noise more efficiently and retain high-quality single-cell data with greater ease.
Summary: If your primary goal is robust empty droplet removal and you seek a cleaner dataset than what current R-based solutions provide, this tool is an excellent choice. However, please note that additional steps will be required to filter out broken cells.
2. Robust Performance on Single-Nucleus Data (snRNA-seq)
This tool demonstrates consistent robustness when applied to single-nucleus RNA sequencing (snRNA-seq) datasets.
Unlike some tools optimized strictly for whole-cell data, our algorithm effectively adapts to the unique count distributions of nuclei.
It successfully identifies empty droplets in snRNA-seq experiments, ensuring high-quality input data for downstream nucleus-specific analyses.
Summary: Whether working with single-cell (scRNA-seq) or single-nucleus (snRNA-seq) data, this tool is an excellent choice for robust empty droplet removal. However, please note that additional steps will be required to filter out broken cells or damaged nuclei.


```python
import dropletqcpy as dp
import pandas as pd
import scanpy as sc

adata = sc.read_10x_mtx(
    "/data_result/dengys/git/dropletqcpy/testdata/10x_mtx/",
    var_names="gene_symbols",  # 或 "gene_ids"
    cache=False
)
df1  = pd.read_csv("/data_result/dengys/git/dropletqcpy/testdata/10x_mtx/metadata.csv", index_col=0)
df2 = pd.read_csv("/data_result/dengys/git/dropletqcpy/testdata/10x_mtx/meta_cellperdict.csv", index_col=0)


# 确保 index 都是字符串
adata.obs.index = adata.obs.index.astype(str)
df1.index = df1.index.astype(str)
df2.index = df2.index.astype(str)
# 直接 join（按 index 自动对齐）
adata.obs = adata.obs.join(df1, how="left")
adata.obs = adata.obs.join(df2, how="left")

adata =  dp.compute_nuclear_fraction_from_bam(adata, 
                                  bam_path = "/data_result/dengys/git/dropletqcpy/testdata/possorted_genome_bam.bam",#)
                                  gtf_path = "/data_result/dengys/git/dropletqcpy/testdata/genes.gtf")




# Step 2 — identify empty droplets
adata = dp.identify_empty_droplets(adata)

────────────────────────────────────────────────────────────────────────
  dropletqcpy — identify_empty_droplets  |  GMM 2D Cluster Report
────────────────────────────────────────────────────────────────────────

Key parameters (adjust to tune classification):
┌──────────────────────────────────────────────────────────┐
│  n_components         = 3    (auto-detected)                │
│  hard_filter_threshold= 1.0     (log10 units)               │
│  min_total_counts     = 500       (raw UMI count)            │
│  min_nuclear_fraction = 0.1                                │
└──────────────────────────────────────────────────────────┘

  Per-cluster statistics:

           n_bins  mean_total_counts  median_total_counts  mean_log10_total_counts  mean_nuclear_fraction  median_nuclear_fraction  min_nuclear_fraction  max_nuclear_fraction  quality
cluster                                                                                                                                                                                
0           15262          3720.9123               2571.5                   3.4431                 0.2328                   0.2345                0.0149                0.4330   ✓  OK 
1            4012            95.6196                 54.0                   1.7797                 0.1601                   0.1709                0.0163                0.4258   ✗  LOW
2            2276          1091.6235                728.5                   2.8598                 0.4691                   0.4613                0.2870                0.7131   ✓  OK 

  ─ Removal summary ─────────────────────────────────────
  Hard-filtered (log10_counts ≤ 1.0)  :   29,609 bins
  Low-quality clusters (empty_droplet) :   33,621 bins  (65.7%)
  Retained (cell)                      :   17,538 bins  (34.3%)
  Total input                          :   51,159 bins
  ────────────────────────────────────────────────────────

  To adjust thresholds, rerun with e.g.:
    identify_empty_droplets(adata,
        n_components=3,       # set None for auto-detect
        min_total_counts=1000,
        min_nuclear_fraction=0.15,
    )
────────────────────────────────────────────────────────────────────────





```










