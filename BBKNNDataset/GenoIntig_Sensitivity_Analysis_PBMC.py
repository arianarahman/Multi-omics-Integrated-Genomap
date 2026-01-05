"""
=============================================================================
Script Name:    GenoIntig_Sensitivity_Analysis_PBMC.py
Author:         Ariana Rahman
Affiliation:    Arizona State University / Stanford University
Date:           January 2026
Description:    
    This script performs a parameter sensitivity analysis for the GenoIntig 
    integration framework. It evaluates the stability of the model against 
    changes in structural hyperparameters (Embedding Dimension and Map Resolution) 
    using the PBMC dataset.

    Key Experiments:
    1. Embedding Dimension Sweep: Testing stability across n_dim=[16, 32, 64, 128].
    2. Map Resolution Sweep: Testing stability across map_size=[30, 40, 50, 60].

    Output:
    - Generates 'Sensitivity_Structural_Results.csv' (raw metrics).
    - Generates 'Parameter_Sensitivity_Plot.png' (visualization for Supplementary Material).

Usage:
    Run this script from the root directory where 'Dataset/' and 'genomap/' 
    folders are located.
    > python GenoIntig_Sensitivity_Analysis_PBMC.py
=============================================================================
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import scanorama
from sklearn.metrics import adjusted_rand_score, silhouette_score
import genomap.genoDR as gp
import anndata as ad
import scipy.sparse

# --- CONFIGURATION ---
OUTPUT_FOLDER = './Sensitivity_Results'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

DATA_FILE = './Dataset/pbmcs_ctrl_annotated.h5ad'
BATCH_KEY = 'batch'
CELLTYPE_KEY = 'celltype'

# --- HELPER FUNCTIONS ---
def load_and_prepare_data():
    """Loads PBMC data and runs initial Scanorama correction."""
    logging.info(f"Loading data from {DATA_FILE}...")
    try:
        adata = sc.read(DATA_FILE)
    except FileNotFoundError:
        logging.error(f"Data file not found at {DATA_FILE}")
        return None

    # Preprocessing
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

    # Scanorama Prep
    if BATCH_KEY not in adata.obs:
         logging.error(f"Batch key '{BATCH_KEY}' not found.")
         return None
         
    adatas_list = [adata[adata.obs[BATCH_KEY] == batch].copy() for batch in adata.obs[BATCH_KEY].unique()]

    logging.info("Running Scanorama...")
    adatas_corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True, hvg=2000)
    adata_full = ad.concat(adatas_corrected, join='outer', label="batch_id")
    
    # Map Labels
    if CELLTYPE_KEY in adata.obs:
        # Re-map labels carefully using index matching
        adata_full.obs['class'] = adata.obs.loc[adata_full.obs_names, CELLTYPE_KEY].values
    else:
        return None
    
    return adata_full

def run_experiment(adata, params, seed):
    """Runs GenoIntig with supported parameters (n_dim, map_size)."""
    np.random.seed(seed)
    
    n_dim = params.get('n_dim', 32)
    map_size = params.get('map_size', 40)
    # Note: We removed 'gamma' since we cannot modify genoDR
    
    n_clusters = len(np.unique(adata.obs['class']))

    try:
        # Standard Call using existing arguments
        resDR = gp.genoDR(
            adata.obsm['X_scanorama'], 
            n_dim=n_dim, 
            n_clusters=n_clusters, 
            colNum=map_size, 
            rowNum=map_size
        )
        
        adata_run = adata.copy()
        adata_run.obsm['X_genointig'] = resDR
        
        sc.pp.neighbors(adata_run, use_rep='X_genointig', n_neighbors=15)
        sc.tl.leiden(adata_run, resolution=0.5)
        
        ari = adjusted_rand_score(adata_run.obs['class'], adata_run.obs['leiden'])
        sil = silhouette_score(adata_run.obsm['X_genointig'], adata_run.obs['leiden'])
        
        return ari, sil
    except Exception as e:
        logging.error(f"Run failed for {params}: {e}")
        return np.nan, np.nan

# --- MAIN SWEEP LOGIC ---
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    adata = load_and_prepare_data()
    if adata is None: return
    
    # 1. DEFINE SWEEPS (Only Structural Parameters)
    sweeps = {
        "Embedding_Dim":   {'param': 'n_dim',    'values': [16, 32, 64, 128], 'defaults': {'map_size': 40}},
        "Map_Resolution":  {'param': 'map_size', 'values': [30, 40, 50, 60],  'defaults': {'n_dim': 32}}
    }
    
    seeds = [42, 123, 2024] 
    all_results = []

    # 2. RUN LOOPS
    for sweep_name, config in sweeps.items():
        logging.info(f"--- Starting Sweep: {sweep_name} ---")
        param_name = config['param']
        
        for val in config['values']:
            for seed in seeds:
                current_params = config['defaults'].copy()
                current_params[param_name] = val
                
                logging.info(f"Running {sweep_name}={val}, Seed={seed}")
                ari, sil = run_experiment(adata, current_params, seed)
                
                all_results.append({
                    "Sweep_Type": sweep_name,
                    "Parameter_Value": val,
                    "Seed": seed,
                    "ARI": ari,
                    "Silhouette": sil
                })

    # 3. SAVE RESULTS
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUTPUT_FOLDER, "Sensitivity_Structural_Results.csv"), index=False)
    
    # 4. PLOT (2 Subplots)
    if not df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # Changed to 2 columns
        
        for i, (sweep_name, ax) in enumerate(zip(sweeps.keys(), axes)):
            subset = df[df["Sweep_Type"] == sweep_name]
            summary = subset.groupby("Parameter_Value")["ARI"].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(summary["Parameter_Value"], summary["mean"], yerr=summary["std"], 
                        fmt='-o', capsize=5, color='navy', label='ARI')
            
            ax.set_title(f"Sensitivity to {sweep_name.replace('_', ' ')}")
            ax.set_xlabel(sweep_name)
            ax.set_ylabel("ARI Score")
            ax.grid(True, linestyle='--', alpha=0.6)
            
            if isinstance(summary["Parameter_Value"].iloc[0], (int, float)):
                 ax.set_xticks(summary["Parameter_Value"])

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "Parameter_Sensitivity_Plot.png"), dpi=300)
        logging.info("Sensitivity analysis complete.")
    else:
        logging.error("No results generated.")

if __name__ == "__main__":
    main()