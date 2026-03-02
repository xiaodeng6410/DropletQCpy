"""
Basic smoke tests for dropletqcpy.

These tests use synthetic data and do not require real BAM or loom files.
They verify that the API contract (inputs → adata.obs keys) is preserved.
"""

import numpy as np
import pytest
import anndata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_adata():
    """Minimal AnnData with nuclear fraction and UMI already in obs."""
    rng = np.random.default_rng(42)
    n_cells = 300

    # Simulate a bimodal NF distribution: cells (~0.1) and empty (~0.8)
    nf = np.concatenate([
        rng.normal(0.1, 0.05, 220).clip(0, 1),   # intact cells
        rng.normal(0.8, 0.05,  80).clip(0, 1),   # empty droplets
    ])
    umi = np.concatenate([
        rng.integers(1000, 8000, 220).astype(float),
        rng.integers(  50,  300,  80).astype(float),
    ])

    # Dummy count matrix (required for adata.X.sum fallback)
    X = rng.poisson(1.0, size=(n_cells, 50)).astype(float)

    obs = {
        "nuclear_fraction": nf,
        "total_umi": umi,
    }
    import pandas as pd
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(obs),
    )
    return adata


@pytest.fixture
def labelled_adata(synthetic_adata):
    """AnnData with empty droplet labels and fake cell type annotations."""
    from dropletqcpy.empty_droplets import identify_empty_droplets
    import pandas as pd

    adata = identify_empty_droplets(
        synthetic_adata,
        umi_key="total_umi",
        fallback_cutoff=0.5,
    )

    # Add fake cell type annotations for the cell subset
    rng = np.random.default_rng(0)
    types = np.where(
        adata.obs["droplet_label"] == "cell",
        rng.choice(["TypeA", "TypeB"], size=adata.n_obs),
        "empty",
    )
    adata.obs["cell_type"] = pd.Categorical(types)
    return adata


# ---------------------------------------------------------------------------
# identify_empty_droplets
# ---------------------------------------------------------------------------

class TestIdentifyEmptyDroplets:

    def test_obs_key_written(self, synthetic_adata):
        from dropletqcpy.empty_droplets import identify_empty_droplets
        adata = identify_empty_droplets(
            synthetic_adata, umi_key="total_umi", fallback_cutoff=0.5
        )
        assert "droplet_label" in adata.obs.columns

    def test_valid_labels_only(self, synthetic_adata):
        from dropletqcpy.empty_droplets import identify_empty_droplets
        adata = identify_empty_droplets(
            synthetic_adata, umi_key="total_umi", fallback_cutoff=0.5
        )
        valid = {"cell", "empty_droplet", None}
        unique = set(adata.obs["droplet_label"].unique())
        assert unique <= valid

    def test_uns_metadata_written(self, synthetic_adata):
        from dropletqcpy.empty_droplets import identify_empty_droplets
        adata = identify_empty_droplets(
            synthetic_adata, umi_key="total_umi", fallback_cutoff=0.5
        )
        assert "dropletqcpy" in adata.uns
        assert "empty_droplets" in adata.uns["dropletqcpy"]
        assert "nf_cutoff" in adata.uns["dropletqcpy"]["empty_droplets"]

    def test_returns_adata(self, synthetic_adata):
        from dropletqcpy.empty_droplets import identify_empty_droplets
        result = identify_empty_droplets(
            synthetic_adata, umi_key="total_umi", fallback_cutoff=0.5
        )
        assert isinstance(result, anndata.AnnData)

    def test_missing_nf_key_raises(self, synthetic_adata):
        from dropletqcpy.empty_droplets import identify_empty_droplets
        with pytest.raises(KeyError):
            identify_empty_droplets(synthetic_adata, nf_key="nonexistent")

    def test_invalid_nf_rescue_raises(self, synthetic_adata):
        from dropletqcpy.empty_droplets import identify_empty_droplets
        with pytest.raises(ValueError):
            identify_empty_droplets(synthetic_adata, nf_rescue=1.5)


# ---------------------------------------------------------------------------
# identify_damaged_cells
# ---------------------------------------------------------------------------

class TestIdentifyDamagedCells:

    def test_obs_key_written(self, labelled_adata):
        from dropletqcpy.damaged_cells import identify_damaged_cells
        adata = identify_damaged_cells(
            labelled_adata, cell_type_key="cell_type", umi_key="total_umi"
        )
        assert "cell_status" in adata.obs.columns

    def test_valid_labels_only(self, labelled_adata):
        from dropletqcpy.damaged_cells import identify_damaged_cells
        adata = identify_damaged_cells(
            labelled_adata, cell_type_key="cell_type", umi_key="total_umi"
        )
        valid = {"cell", "empty_droplet", "damaged_cell"}
        unique = set(adata.obs["cell_status"].unique())
        assert unique <= valid

    def test_empty_droplets_preserved(self, labelled_adata):
        from dropletqcpy.damaged_cells import identify_damaged_cells
        adata = identify_damaged_cells(
            labelled_adata, cell_type_key="cell_type", umi_key="total_umi"
        )
        orig_empty = labelled_adata.obs["droplet_label"] == "empty_droplet"
        new_status = adata.obs.loc[orig_empty, "cell_status"]
        assert (new_status == "empty_droplet").all()

    def test_uns_metadata_written(self, labelled_adata):
        from dropletqcpy.damaged_cells import identify_damaged_cells
        adata = identify_damaged_cells(
            labelled_adata, cell_type_key="cell_type", umi_key="total_umi"
        )
        assert "damaged_cells" in adata.uns["dropletqcpy"]
        assert "per_cell_type" in adata.uns["dropletqcpy"]["damaged_cells"]

    def test_returns_adata(self, labelled_adata):
        from dropletqcpy.damaged_cells import identify_damaged_cells
        result = identify_damaged_cells(
            labelled_adata, cell_type_key="cell_type", umi_key="total_umi"
        )
        assert isinstance(result, anndata.AnnData)

    def test_missing_cell_type_key_raises(self, labelled_adata):
        from dropletqcpy.damaged_cells import identify_damaged_cells
        with pytest.raises(KeyError):
            identify_damaged_cells(labelled_adata, cell_type_key="nonexistent")
