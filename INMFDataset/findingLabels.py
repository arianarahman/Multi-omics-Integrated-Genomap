import scanpy as sc
import pandas as pd
DATA_FILE = './Dataset/pbmcs_ctrl_converted.h5ad'
adata = sc.read_h5ad(DATA_FILE)

# Standard preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable].copy()
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=30, metric="cosine")
sc.tl.leiden(adata, resolution=0.7)
sc.tl.umap(adata)

# Canonical PBMC markers (robust, not exhaustive)
markers = {
    "T":        ["CD3D","CD3E","TRAC","IL7R","CCR7","S100A4"],
    "NK":       ["NKG7","GNLY","PRF1","GZMB"],
    "B":        ["MS4A1","CD79A","CD74","IGKC","IGHM"],
    "Mono_C":   ["LYZ","S100A8","S100A9","LST1"],
    "Mono_NC":  ["FCGR3A","MS4A7","LILRB1"],
    "DC":       ["FCER1A","CST3","LILRA4"],
    "Platelet": ["PPBP","PF4"]
}

# Score each cell for each marker set
for lab, genes in markers.items():
    genes = [g for g in genes if g in adata.var_names]  # keep genes present
    sc.tl.score_genes(adata, gene_list=genes, score_name=f"score_{lab}")

# Assign a per-cell label by the highest score
score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
adata.obs["celltype_pred"] = (
    adata.obs[score_cols].idxmax(axis=1).str.replace("score_","", regex=False)
)

# (Optional) smooth to cluster majorities
majority = adata.obs.groupby("leiden")["celltype_pred"].agg(lambda s: s.value_counts().idxmax())
adata.obs["celltype"] = adata.obs["leiden"].map(majority)

# Save/inspect
adata.write("./Dataset/pbmcs_ctrl_annotated.h5ad")
print(adata.obs["celltype"].value_counts())
