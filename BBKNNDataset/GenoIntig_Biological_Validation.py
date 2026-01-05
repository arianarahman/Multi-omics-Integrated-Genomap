"""
=============================================================================
Script Name:    GenoIntig_Biological_Validation.py
Author:         Ariana Rahman
Affiliation:    Arizona State University / Stanford University
Date:           January 2026
Description:    
    This script performs biological validation of the GenoIntig integration 
    framework. It verifies that the integration process preserves biological 
    signals by examining the expression patterns of canonical marker genes.

    Fixes applied:
    - Added NaN cleaning to prevent numerical errors.
    - Removed 'rotation' argument from stacked_violin to fix matplotlib crash.
=============================================================================
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scanorama
import anndata as ad
import scipy.sparse
import genomap.genoDR as gp

# --- Configuration ---
DATA_FILE = './Dataset/pbmcs_ctrl_annotated.h5ad'
OUTPUT_FOLDER = './Biological_Validation_Results'
BATCH_KEY = 'batch'
CELLTYPE_KEY = 'celltype'

# --- Canonical Markers Definition ---
MARKER_DICT = {
    'T cells': ['CD3D', 'CD3E'],
    'B cells': ['MS4A1'],
    'Monocytes': ['LST1', 'S100A8']
}
FLAT_MARKERS = [gene for sublist in MARKER_DICT.values() for gene in sublist]

def check_and_fix_nans(adata, stage="Loading"):
    """Helper to check for and replace NaNs/Infs in the data matrix."""
    if scipy.sparse.issparse(adata.X):
        if np.isnan(adata.X.data).any() or np.isinf(adata.X.data).any():
            logging.warning(f"NaNs or Infs found in sparse matrix during {stage}. Replacing with 0.")
            adata.X.data = np.nan_to_num(adata.X.data, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        if np.isnan(adata.X).any() or np.isinf(adata.X).any():
            logging.warning(f"NaNs or Infs found in dense matrix during {stage}. Replacing with 0.")
            adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
    return adata

def load_and_process_data():
    """Loads and performs standard normalization for visualization."""
    if not os.path.exists(DATA_FILE):
        logging.error(f"Data file not found: {DATA_FILE}")
        return None
    
    adata = sc.read(DATA_FILE)
    adata.var_names_make_unique()
    
    # 1. Clean Data immediately after loading
    adata = check_and_fix_nans(adata, stage="Initial Load")

    # 2. Standard normalization
    sc.pp.filter_cells(adata, min_genes=1)
    if adata.n_obs == 0:
        logging.error("No cells remained after filtering!")
        return None
        
    sc.pp.normalize_total(adata, target_sum=1e4)
    adata = check_and_fix_nans(adata, stage="Pre-Log1p")
    
    sc.pp.log1p(adata)
    adata = check_and_fix_nans(adata, stage="Post-Log1p")
    
    adata.raw = adata
    return adata

def run_genointig_pipeline(adata):
    """Runs the full GenoIntig pipeline to get the 'X_genomap' embedding."""
    logging.info("Running GenoIntig Pipeline...")
    
    if BATCH_KEY not in adata.obs:
        logging.error(f"Batch key {BATCH_KEY} not found.")
        return adata

    adatas_list = [adata[adata.obs[BATCH_KEY] == batch].copy() for batch in adata.obs[BATCH_KEY].unique()]
    
    # Scanorama
    corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True, hvg=2000)
    adata_int = ad.concat(corrected, join='outer', label="post_scanorama_batch_id")
    
    if adata.n_obs == adata_int.n_obs:
        adata.obsm['X_scanorama'] = adata_int.obsm['X_scanorama']
    else:
        logging.warning("Shape mismatch after Scanorama. Using concatenated object.")
        adata = adata_int

    # GenoMap
    n_clusters = 8 
    if CELLTYPE_KEY in adata.obs:
         n_clusters = len(adata.obs[CELLTYPE_KEY].unique())

    if np.isnan(adata.obsm['X_scanorama']).any():
         logging.warning("NaNs found in Scanorama embedding. Cleaning...")
         adata.obsm['X_scanorama'] = np.nan_to_num(adata.obsm['X_scanorama'])

    adata.obsm['X_genomap'] = gp.genoDR(
        adata.obsm['X_scanorama'], 
        n_dim=32, 
        n_clusters=n_clusters, 
        colNum=33, 
        rowNum=33
    )
    
    # Clustering
    logging.info("Clustering on GenoIntig embedding...")
    sc.pp.neighbors(adata, use_rep='X_genomap', n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.5, key_added='genointig_clusters')
    
    return adata

def generate_biological_plots(adata):
    """Generates Figure 7: DotPlot and Violin Plots of canonical markers."""
    logging.info("Generating Biological Signal Plots...")
    
    valid_markers = {k: [g for g in v if g in adata.var_names] for k, v in MARKER_DICT.items()}
    
    # 1. Dot Plot
    plt.figure(figsize=(10, 6))
    sc.pl.dotplot(adata, valid_markers, groupby='genointig_clusters', 
                  standard_scale='var', show=False, title="Marker Expression in GenoIntig Clusters")
    
    save_path = os.path.join(OUTPUT_FOLDER, "Figure_7a_Dotplot_Clusters.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved Dotplot to {save_path}")
    
    # 2. Violin Plot (Fixed Crash)
    plt.figure(figsize=(12, 6))
    
    # --- FIX: Removed 'rotation=45' to prevent AttributeError ---
    sc.pl.stacked_violin(adata, valid_markers, groupby=CELLTYPE_KEY, 
                         show=False, title="Marker Preservation by Cell Type")
    
    # Apply rotation manually to the figure if needed
    plt.xticks(rotation=45)

    save_path = os.path.join(OUTPUT_FOLDER, "Figure_7b_Violin_CellTypes.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved Violin Plot to {save_path}")

def check_de_consistency(adata):
    """Generates Table IV: Verifies marker consistency."""
    logging.info("Running Differential Expression (Consistency Check)...")
    
    try:
        sc.tl.rank_genes_groups(adata, 'genointig_clusters', method='t-test')
    except Exception as e:
        logging.error(f"DE Analysis failed: {e}")
        return

    result_df = sc.get.rank_genes_groups_df(adata, group=None)
    consistency_data = []
    
    for cell_type, markers in MARKER_DICT.items():
        for gene in markers:
            if gene not in adata.var_names:
                continue
                
            gene_rows = result_df[result_df['names'] == gene].sort_values('scores', ascending=False)
            
            if not gene_rows.empty:
                best_match = gene_rows.iloc[0]
                cluster_id = best_match['group']
                cluster_genes = result_df[result_df['group'] == cluster_id]['names'].tolist()
                rank = cluster_genes.index(gene) if gene in cluster_genes else 999
                score = best_match['scores']
                
                consistency_data.append({
                    "Cell Type Group": cell_type,
                    "Marker Gene": gene,
                    "Best Mapping Cluster": cluster_id,
                    "Rank in Cluster": rank + 1,
                    "Z-Score": f"{score:.2f}",
                    "Status": "Preserved" if rank < 100 else "Lost"
                })
    
    df_consistency = pd.DataFrame(consistency_data)
    csv_path = os.path.join(OUTPUT_FOLDER, "Table_IV_DE_Consistency.csv")
    df_consistency.to_csv(csv_path, index=False)
    
    logging.info(f"Table IV saved to {csv_path}")
    print("\n--- Table IV: Marker Gene Consistency ---")
    print(df_consistency)

def main():
    sc.settings.verbosity = 0
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    adata = load_and_process_data()
    if adata is None: return
    
    adata = run_genointig_pipeline(adata)
    
    generate_biological_plots(adata)
    check_de_consistency(adata)

if __name__ == "__main__":
    main()