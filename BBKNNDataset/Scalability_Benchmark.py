"""
=============================================================================
Script Name:    Scalability_Benchmark.py
Author:         Ariana Rahman
Affiliation:    Arizona State University / Stanford University
Date:           January 2026
Description:    
    This script implements a unified test harness to benchmark the computational 
    efficiency (Runtime and Peak Memory) of the GenoIntig integration framework 
    against baseline methods (Scanorama, BBKNN, iNMF).

    Key Experiments:
    1. Runtime Scalability: Measuring execution time (seconds) across increasing 
       dataset sizes (10k, 25k, 50k, 100k cells).
    2. Memory Scalability: Measuring peak RAM usage (MB) using tracemalloc 
       across the same dataset sizes.

    Output:
    - Generates 'Scalability_Results.csv' (raw metrics).
    - Generates 'Figure_Scalability.png' (visualization for the manuscript).

Usage:
    Run this script from the root directory where 'Dataset/' and 'genomap/' 
    folders are located.
    > python Scalability_Benchmark.py
=============================================================================
"""

import os
import time
import logging
import gc
import tracemalloc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import scanorama
import bbknn
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import anndata as ad
import scipy.sparse
import genomap.genoDR as gp

# This script implements a unified test harness that measures Runtime (seconds) and Peak Memory (MB) for GenoIntig against the 
# baseline methods (Scanorama, BBKNN, iNMF) across increasing dataset sizes (10k, 25k, 50k, 100k).

# --- Configuration ---
DATA_FILE = './Dataset/pbmcs_ctrl_annotated.h5ad'
OUTPUT_FOLDER = './Scalability_Results'
BATCH_KEY = 'batch'
CELLTYPE_KEY = 'celltype'

# Define the subsample sizes for the experiment
# Note: If your dataset is smaller than these numbers, the script will sample with replacement.
SAMPLE_SIZES = [10000, 25000, 50000, 100000]

# --- Methods Wrappers ---
# Each wrapper performs the specific integration task and returns nothing
# The harness handles timing and memory tracking.

def run_inmf(adata):
    """Ref: iNMF_Algorithom_iNMF_Dataset.py"""
    # Shift to non-negative
    if scipy.sparse.issparse(adata.X):
        min_val = adata.X.data.min()
        if min_val < 0:
            adata.X.data -= min_val
    else:
        min_val = adata.X.min()
        if min_val < 0:
            adata.X -= min_val
            
    model = NMF(n_components=30, init='nndsvda', random_state=42)
    adata.obsm['X_nmf'] = model.fit_transform(adata.X)
    return adata

def run_bbknn(adata):
    """Ref: BBKNN_Algorithom_iNMF_Dataset.py"""
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
    bbknn.bbknn(adata, batch_key=BATCH_KEY)
    return adata

def run_scanorama(adata):
    """Ref: Scanorama_Algorithom_iNMF_Datase.py"""
    # Scanorama requires list of adatas
    adatas_list = [adata[adata.obs[BATCH_KEY] == batch].copy() for batch in adata.obs[BATCH_KEY].unique()]
    corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True, hvg=2000)
    adata_int = ad.concat(corrected, join='outer', label="post_scanorama_batch_id")
    return adata_int

def run_genointig(adata):
    """Ref: Genomap_Algorithom_iNMF_Dataset.py"""
    # Stage 1: Scanorama (GenoIntig uses Scanorama as precursor)
    adatas_list = [adata[adata.obs[BATCH_KEY] == batch].copy() for batch in adata.obs[BATCH_KEY].unique()]
    corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True, hvg=2000)
    adata_int = ad.concat(corrected, join='outer', label="post_scanorama_batch_id")
    
    # Stage 2: GenoMap (genoDR)
    # Using parameters from your script
    n_clusters = 8 
    adata_int.obsm['X_genomap'] = gp.genoDR(
        adata_int.obsm['X_scanorama'], 
        n_dim=32, 
        n_clusters=n_clusters, 
        colNum=33, 
        rowNum=33
    )
    return adata_int

# Dictionary mapping names to functions
METHODS = {
    "iNMF": run_inmf,
    "BBKNN": run_bbknn,
    "Scanorama": run_scanorama,
    "GenoIntig": run_genointig
}

# --- Utilities ---

def load_base_data():
    """Loads and performs minimal common preprocessing."""
    if not os.path.exists(DATA_FILE):
        logging.error(f"Data file not found: {DATA_FILE}")
        return None
    
    adata = sc.read(DATA_FILE)
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Handle NaNs generically
    if scipy.sparse.issparse(adata.X):
        if np.isnan(adata.X.data).any():
            adata.X.data[np.isnan(adata.X.data)] = 0
    elif np.issubdtype(adata.X.dtype, np.floating):
        if np.isnan(adata.X).any():
            adata.X[np.isnan(adata.X)] = 0
            
    return adata

def get_subsampled_data(adata_full, n_cells):
    """Subsamples (or upsamples with replacement) to reach target size."""
    if n_cells > adata_full.n_obs:
        logging.warning(f"Target {n_cells} > Actual {adata_full.n_obs}. Upsampling with replacement.")
        # Random choice with replacement to simulate larger dataset
        indices = np.random.choice(adata_full.n_obs, size=n_cells, replace=True)
        return adata_full[indices].copy()
    else:
        return sc.pp.subsample(adata_full, n_obs=n_cells, copy=True, random_state=42)

def benchmark_method(method_name, adata_input):
    """Runs a method and tracks time and peak memory."""
    # Force garbage collection before start
    gc.collect()
    
    # Start tracking
    tracemalloc.start()
    start_time = time.time()
    
    try:
        # Run the specific pipeline
        METHODS[method_name](adata_input)
        status = "Success"
    except Exception as e:
        logging.error(f"Error running {method_name}: {e}")
        status = "Failed"
        
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    runtime = end_time - start_time
    peak_mb = peak / (1024 * 1024) # Convert bytes to MB
    
    return runtime, peak_mb, status

def plot_scalability(df):
    """Generates Figure 6: Side-by-side Scaling Curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Runtime
    sns.lineplot(data=df, x="Cells", y="Runtime (s)", hue="Method", style="Method", 
                 markers=True, dashes=False, linewidth=2.5, ax=axes[0])
    axes[0].set_title("Runtime Scalability", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_xlabel("Number of Cells")
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Memory
    sns.lineplot(data=df, x="Cells", y="Peak Memory (MB)", hue="Method", style="Method", 
                 markers=True, dashes=False, linewidth=2.5, ax=axes[1])
    axes[1].set_title("Memory Scalability", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Peak RAM Usage (MB)")
    axes[1].set_xlabel("Number of Cells")
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_FOLDER, "Figure_Scalability.png")
    plt.savefig(save_path, dpi=600)
    logging.info(f"Figure  saved to {save_path}")

def main():
    sc.settings.verbosity = 0
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    logging.info("Loading master dataset...")
    adata_full = load_base_data()
    if adata_full is None: return

    results = []

    # --- Benchmarking Loop ---
    for n_cells in SAMPLE_SIZES:
        logging.info(f"--- Benchmarking Dataset Size: {n_cells} cells ---")
        
        # Create fresh subsample for this size iteration
        adata_sub = get_subsampled_data(adata_full, n_cells)
        
        for method in METHODS.keys():
            logging.info(f"Running {method} on {n_cells} cells...")
            
            # Pass a copy so one method doesn't pollute the object for the next
            adata_run = adata_sub.copy()
            
            runtime, peak_mem, status = benchmark_method(method, adata_run)
            
            if status == "Success":
                results.append({
                    "Method": method,
                    "Cells": n_cells,
                    "Runtime (s)": runtime,
                    "Peak Memory (MB)": peak_mem
                })
                logging.info(f"  -> Time: {runtime:.2f}s | RAM: {peak_mem:.2f} MB")
            
            del adata_run
        
        del adata_sub
        gc.collect()

    # --- Save Results ---
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_FOLDER, "Scalability_Results.csv")
    df_results.to_csv(csv_path, index=False)
    logging.info(f"Raw data saved to {csv_path}")
    
    if not df_results.empty:
        plot_scalability(df_results)
        
        # Print Table for Paper (Latex style preview)
        print("\n--- Summary Table (Mean Runtime/Mem) ---")
        print(df_results.pivot(index="Cells", columns="Method", values=["Runtime (s)", "Peak Memory (MB)"]))

if __name__ == "__main__":
    main()