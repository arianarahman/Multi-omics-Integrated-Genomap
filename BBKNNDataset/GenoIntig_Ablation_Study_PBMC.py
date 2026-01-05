"""
=============================================================================
Script Name:    GenoIntig_Ablation_Study_PBMC.py
Author:         Ariana Rahman
Affiliation:    Arizona State University / Stanford University
Date:           January 2026
Description:    
    This script performs the Ablation Study for GenoIntig by selectively 
    disabling key components (Spatial Mapping and Clustering Loss) within 
    the actual model training loop.

    Variants Evaluated:
    1. Full GenoIntig: Standard model.
    2. No 2D Mapping: Disables the spatial transformation (Stage 1).
    3. No Clustering Loss: Sets clustering loss weight (gamma) to 0.
    4. Autoencoder Only: Disables both spatial map and clustering loss.

    Output:
    - Generates 'Ablation_Raw_Data.csv' (raw metrics).
    - Generates 'Figure_5_Ablation_Comparison.png' (Visual bar chart).
    - Generates 'Table_III_Ablation_Results.csv' (Mean ± Std table).

Usage:
    Run this script from the root directory. Ensure genomap.genoDR is updated 
    to accept 'gamma' and 'use_spatial_mapping' arguments.
    > python GenoIntig_Ablation_Study_PBMC.py
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
from sklearn.metrics import adjusted_rand_score, silhouette_score, rand_score
import anndata as ad
import scipy.sparse
import genomap.genoDR as gp

# --- Configuration ---
DATA_FILE = './Dataset/pbmcs_ctrl_annotated.h5ad'
OUTPUT_FOLDER = './Ablation_Results'
BATCH_KEY = 'batch'
CELLTYPE_KEY = 'celltype'
SEEDS = [42, 123, 2021]  # Multiple seeds for robust statistics

# --- Variants Definition ---
# These parameters must be supported by your updated gp.genoDR function
VARIANTS = {
    "Full GenoIntig":       {"gamma": 1.0, "use_spatial": True},
    "No 2D Mapping":        {"gamma": 1.0, "use_spatial": False}, 
    "No Clustering Loss":   {"gamma": 0.0, "use_spatial": True},  
    "AE Only (Baseline)":   {"gamma": 0.0, "use_spatial": False}  
}

def load_data():
    """Loads and preprocesses data (Standardized)."""
    try:
        adata = sc.read(DATA_FILE)
    except FileNotFoundError:
        logging.error(f"Data file not found at {DATA_FILE}")
        return None
        
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Handle NaNs
    if scipy.sparse.issparse(adata.X):
        if np.isnan(adata.X.data).any():
            adata.X.data[np.isnan(adata.X.data)] = 0
    elif np.issubdtype(adata.X.dtype, np.floating):
        if np.isnan(adata.X).any():
            adata.X[np.isnan(adata.X)] = 0
    
    return adata

def run_pipeline(adata_raw, variant_name, params, seed):
    """
    Runs a single instance of the pipeline for a specific variant and seed.
    """
    logging.info(f"--- Running {variant_name} (Seed: {seed}) ---")
    
    # 1. Batch Correction (Scanorama)
    # We re-run this to ensure a clean state for every seed/variant
    adatas_list = [adata_raw[adata_raw.obs[BATCH_KEY] == batch].copy() for batch in adata_raw.obs[BATCH_KEY].unique()]
    corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True, hvg=2000, seed=seed)
    adata = ad.concat(corrected, join='outer', label="post_scanorama_batch_id")
    adata.obs[CELLTYPE_KEY] = adata_raw.obs[CELLTYPE_KEY].copy()
    
    # 2. Apply GenoMap (with Ablation Params)
    n_clusters = 8 
    
    try:
        # Pass ablation parameters to the function
        # NOTE: Ensure your library is updated to accept these!
        adata.obsm['X_genomap'] = gp.genoDR(
            adata.obsm['X_scanorama'], 
            n_dim=32, 
            n_clusters=n_clusters, 
            colNum=33, 
            rowNum=33,
            gamma=params['gamma'],         
            use_spatial_mapping=params['use_spatial'] 
        )
    except TypeError as e:
        logging.error(f"gp.genoDR failed. Did you update the function parameters? Error: {e}")
        raise e

    # 3. Clustering
    sc.pp.neighbors(adata, use_rep='X_genomap', n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.5, random_state=seed)
    
    # 4. Evaluation
    true_labels = adata.obs[CELLTYPE_KEY].values
    predicted_labels = adata.obs['leiden'].values

    metrics = {
        "Variant": variant_name,
        "Seed": seed,
        "ARI": adjusted_rand_score(true_labels, predicted_labels),
        "Rand Index": rand_score(true_labels, predicted_labels),
        "Silhouette": silhouette_score(adata.obsm['X_genomap'], predicted_labels)
    }
    return metrics

def plot_ablation_figure(df_results):
    """Generates Figure 5: Grouped Bar Chart of Metrics."""
    df_melt = df_results.melt(id_vars=["Variant", "Seed"], 
                              value_vars=["ARI", "Rand Index", "Silhouette"], 
                              var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create Bar Plot with Error Bars
    ax = sns.barplot(data=df_melt, x="Metric", y="Score", hue="Variant", 
                     palette="viridis", errorbar="sd", capsize=.1)
    
    plt.title("Figure 5: Ablation Study of GenoIntig Components", fontsize=14, fontweight='bold')
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend(title="Model Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_FOLDER, "Figure_5_Ablation_Comparison.png")
    plt.savefig(save_path, dpi=300)
    logging.info(f"Figure 5 saved to {save_path}")

def generate_table_iii(df_results):
    """Generates Table III: Mean +/- Std deviation CSV."""
    summary = df_results.groupby("Variant")[["ARI", "Rand Index", "Silhouette"]].agg(['mean', 'std'])
    
    table_iii = pd.DataFrame(index=summary.index)
    for metric in ["ARI", "Rand Index", "Silhouette"]:
        table_iii[metric] = summary[metric]['mean'].round(3).astype(str) + " ± " + summary[metric]['std'].round(3).astype(str)
    
    save_path = os.path.join(OUTPUT_FOLDER, "Table_III_Ablation_Results.csv")
    table_iii.to_csv(save_path)
    logging.info(f"Table III saved to {save_path}")
    print("\n--- Table III Preview ---")
    print(table_iii)

def main():
    sc.settings.verbosity = 0
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    logging.info("Loading Data...")
    adata_raw = load_data()
    if adata_raw is None: return

    all_results = []

    # --- Main Experiment Loop ---
    for variant_name, params in VARIANTS.items():
        for seed in SEEDS:
            try:
                metrics = run_pipeline(adata_raw, variant_name, params, seed)
                all_results.append(metrics)
                print(f"Finished {variant_name} (Seed {seed}): ARI={metrics['ARI']:.3f}")
            except Exception as e:
                logging.error(f"Failed run for {variant_name} (Seed {seed}): {e}")

    # Compile Results
    df_results = pd.DataFrame(all_results)
    
    # Save Raw Data
    df_results.to_csv(os.path.join(OUTPUT_FOLDER, "Ablation_Raw_Data.csv"), index=False)

    # Generate Paper Artifacts
    if not df_results.empty:
        plot_ablation_figure(df_results)
        generate_table_iii(df_results)
    else:
        logging.error("No results generated.")

if __name__ == "__main__":
    main()