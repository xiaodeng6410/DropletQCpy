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





#展示一下分群
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,5))
sns.scatterplot(
    x=adata.obs["log1p_total_counts"],
    y=adata.obs["nuclear_fraction"],
    hue=adata.obs["gmm_cluster"],
    palette="tab10",
    s=3,
    linewidth=0,
    alpha=0.6
)
plt.xlabel("log1p_total_counts")
plt.ylabel("nuclear_fraction")
plt.title("GMM Clusters in QC Space")
plt.legend(title="Cluster", bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.savefig("QC_GMM_clusters_scRNA.png", dpi=300, bbox_inches="tight")
plt.close()


#测试一下loom方案
#如果loom是用velocyto计算得到的数据，这些细胞是cellranger估算后得到的barcode，
#这也导致细胞数目少于正常的原始数据，所以会出现NaN的情况，在计算前可以考虑删除掉这群细胞，
#显然loom因为体积小比BAM更快，所以先计算loom的核质比，再把结果合并到adata里是个不错的选择，后续我们会优化BAM的运算速度。
#对于loom的数据
#将Bracde进行转换，去掉样本名前缀和末尾x，得到核心序列；
#同时对adata的barcode也进行转换，去掉-1后缀，得到核心序列。这样就能保证两者的barcode能够正确匹配。
adata = dp.compute_nuclear_fraction(
    adata,
    "/data_result/dengys/git/dropletqcpy/testdata/OES272835.loom",
    barcode_transform=lambda bc: bc.split(":")[-1].rstrip("x"),
    adata_barcode_transform=lambda bc: bc.rsplit("-", 1)[0],
)
adata = adata[~adata.obs["nuclear_fraction"].isna()].copy()
adata = dp.identify_empty_droplets(adata)



────────────────────────────────────────────────────────────────────────
  dropletqcpy — identify_empty_droplets  |  GMM 2D Cluster Report
────────────────────────────────────────────────────────────────────────

Key parameters (adjust to tune classification):
┌──────────────────────────────────────────────────────────┐
│  n_components         = 2    (auto-detected)                │
│  hard_filter_threshold= 1.0     (log10 units)               │
│  min_total_counts     = 500       (raw UMI count)            │
│  min_nuclear_fraction = 0.1                                │
└──────────────────────────────────────────────────────────┘

  Per-cluster statistics:

           n_bins  mean_total_counts  median_total_counts  mean_log10_total_counts  mean_nuclear_fraction  median_nuclear_fraction  min_nuclear_fraction  max_nuclear_fraction  quality
cluster                                                                                                                                                                                
0            1926          1412.4964               1053.5                   3.0406                 0.5138                   0.5020                0.1292                0.7881   ✓  OK 
1           15127          3727.7510               2579.0                   3.4408                 0.2784                   0.2809                0.0224                0.5072   ✓  OK 

  ─ Removal summary ─────────────────────────────────────
  Hard-filtered (log10_counts ≤ 1.0)  :        0 bins
  Low-quality clusters (empty_droplet) :        0 bins  (0.0%)
  Retained (cell)                      :   17,053 bins  (100.0%)
  Total input                          :   17,053 bins
  ────────────────────────────────────────────────────────

  To adjust thresholds, rerun with e.g.:
    identify_empty_droplets(adata,
        n_components=2,       # set None for auto-detect
        min_total_counts=1000,
        min_nuclear_fraction=0.15,
    )
────────────────────────────────────────────────────────────────────────

#展示一下分群
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,5))
sns.scatterplot(
    x=adata.obs["log1p_total_counts"],
    y=adata.obs["nuclear_fraction"],
    hue=adata.obs["gmm_cluster"],
    palette="tab10",
    s=3,
    linewidth=0,
    alpha=0.6
)
plt.xlabel("log1p_total_counts")
plt.ylabel("nuclear_fraction")
plt.title("GMM Clusters in QC Space")
plt.legend(title="Cluster", bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.savefig("QC_GMM_clusters_loom.png", dpi=300, bbox_inches="tight")
plt.close()





```










