"""
=============================================================================
Script Name:    Scanorama_Algorithom_iNMF_Datase.py
Author:         Ariana Rahman
Affiliation:    Arizona State University / Stanford University
Date:           January 2026
Description:    
    This script performs single-cell integration using Scanorama and evaluates 
    its performance on PBMC datasets. It includes advanced benchmarking modules 
    for Batch Mixing Assessment and Robustness to Dataset Imbalance.

    Key Experiments:
    1. Full Integration & Visualization: Generates UMAP/t-SNE plots and calculates 
       standard clustering metrics (ARI, Silhouette, Rand Index).
    2. Batch Mixing Assessment: Calculates iLISI (Integration LISI) and kBET 
       to quantify how well batches are mixed within cell types.
    3. Robustness Analysis: Systematically downsamples the largest batch (100%, 
       50%, 25%, 10%) to evaluate stability of performance metrics under imbalance.

    Output:
    - Generates 'UMAP/TSNE' plots for visual inspection.
    - Generates 'CSV' files with ARI, Silhouette, iLISI, and kBET scores.
    - Generates 'Scanorama_Imbalance_Robustness.png' showing metric stability.

Usage:
    Run this script from the root directory where 'Dataset/' folder is located.
    > python Scanorama_Algorithom_iNMF_Datase.py
=============================================================================
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score, rand_score
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import scanorama
import anndata as ad
import scipy.sparse
from scib.metrics import kBET

# --- Configuration ---
DATA_FILE = './Dataset/pbmcs_ctrl_annotated.h5ad'
OUTPUT_FOLDER = './Figures_CSVs'
BATCH_KEY = 'batch'
CELLTYPE_KEY = 'celltype'

# --- Control Flags ---
CALCULATE_BATCH_METRICS = True   # Set to False to skip iLISI/kBET if running quickly
RUN_ROBUSTNESS_ANALYSIS = True   # Set to False to skip the slow downsampling loop

# --- Helper Functions (Standardized) ---

def create_custom_colormap(labels, colors):
    """
    Creates a custom color mapping dictionary and a list of legend patches.
    """
    label_to_color = dict(zip(labels, colors))
    legend_handles = [Patch(color=label_to_color[label], label=label) for label in labels]
    return label_to_color, legend_handles

def plot_embedding(X, y, title, filename, custom_colors=None):
    X, y = np.asarray(X), np.asarray(y)
    plt.figure()

    if custom_colors:
        unique_labels = np.unique(y)
        for label in unique_labels:
            indices = np.where(y == label)
            plt.scatter(X[indices, 0], X[indices, 1],
                        color=custom_colors.get(label),
                        label=label,
                        s=18)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet', s=18)

    plt.xlabel(f'{title} Dimension 1', fontsize=15)
    plt.ylabel(f'{title} Dimension 2', fontsize=15)
    plt.title(filename, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{filename}.png'), dpi=600)
    plt.close()

def plot_metrics_bar(ari, rand, silhouette, filename_prefix):
    """Generates and saves a bar chart of evaluation metrics."""
    metrics = ["Adjusted Rand Index (ARI)", "Rand Index", "Silhouette Score"]
    values = [ari, rand, silhouette]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['coral', 'lightgreen', 'blue'], width=0.3)
    plt.ylabel("Score")
    plt.title(f"ARI_RI_SC_Metrics_For_{filename_prefix}")
    plt.ylim(min(0, min(values) - 0.1), max(1, max(values) + 0.1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.2f}', ha='center',
                 color='black', fontsize=12, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"ARI_RI_SC_Metrics_For_{filename_prefix}.png"), dpi=300)
    plt.close()

def plot_batch_metrics(ilisi, kbet, filename):
    plt.figure(figsize=(6,5))
    metrics = ["iLISI", "kBET"]
    values = [ilisi, kbet]
    bars = plt.bar(metrics, values)
    plt.ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}", ha="center")
    plt.ylabel("Score")
    plt.title("Batch Mixing Performance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"BatchMetrics_{filename}.png"), dpi=300)
    plt.close()

def compute_lisi_python(X, labels, k=90):
    """
    A Python-only implementation of LISI (Local Inverse Simpson's Index).
    """
    n_cells = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    
    unique_labels, encoded_labels = np.unique(labels, return_inverse=True)
    n_classes = len(unique_labels)
    
    lisi_scores = []
    for i in range(n_cells):
        neighbor_indices = indices[i]
        neighbor_classes = encoded_labels[neighbor_indices]
        counts = np.bincount(neighbor_classes, minlength=n_classes)
        probs = counts / k
        simpson = np.sum(probs ** 2)
        lisi = 1 / simpson if simpson > 0 else 1
        lisi_scores.append(lisi)
        
    return np.mean(lisi_scores)

def compute_all_metrics(adata):
    """Computes ARI, Silhouette, iLISI, kBET."""
    ari = 0.0
    if CELLTYPE_KEY in adata.obs:
        ari = adjusted_rand_score(adata.obs[CELLTYPE_KEY], adata.obs['leiden'])
    
    try:
        silhouette = silhouette_score(adata.obsm['X_scanorama'], adata.obs['leiden'])
    except:
        silhouette = 0.0
        
    ilisi = 0.0
    kbet = 0.0
    
    if CALCULATE_BATCH_METRICS:
        try:
            # We use 'X_scanorama' as the integrated embedding for LISI
            ilisi = compute_lisi_python(adata.obsm['X_scanorama'], adata.obs[BATCH_KEY], k=90)
        except Exception as e:
            logging.warning(f"iLISI failed: {e}")

        try:
            # kBET might fail on Windows if R is not installed
            kbet_result = kBET(adata, batch_key=BATCH_KEY, label_key=CELLTYPE_KEY, type_="embed", embed="X_scanorama", verbose=False)
            kbet = np.mean(kbet_result['accept'])
        except Exception as e:
            logging.warning(f"kBET failed (common on Windows without R): {e}")
            kbet = 0.0
        
    return ari, silhouette, ilisi, kbet

def run_non_overlapping_experiment(adata_raw):
    """
    Simulates a non-overlapping scenario by removing a specific cell type 
    from ONE batch only. 
    
    Goal: Verify that the unique cell type in the remaining batches 
    does NOT incorrectly merge with unrelated cells in the modified batch.
    """
    logging.info("--- Starting Non-Overlapping (Negative Control) Experiment ---")
    
    # 1. Setup: Identify a target cell type and batch
    # We will remove 'B cells' (or similar) from the largest batch
    target_cell_type = 'B cells'  # Adjust based on your actual labels (e.g., 'CD4 T cells')
    
    # Check if cell type exists
    if target_cell_type not in adata_raw.obs[CELLTYPE_KEY].values:
        # Fallback: pick the first available cell type
        target_cell_type = adata_raw.obs[CELLTYPE_KEY].unique()[0]
    
    target_batch = adata_raw.obs[BATCH_KEY].value_counts().idxmax()
    
    logging.info(f"Scenario: Removing '{target_cell_type}' from Batch '{target_batch}'...")

    # 2. Create the Imbalanced Dataset
    # Keep cells that are NOT (Target Batch AND Target Cell Type)
    mask = ~((adata_raw.obs[BATCH_KEY] == target_batch) & 
             (adata_raw.obs[CELLTYPE_KEY] == target_cell_type))
    
    adata_neg = adata_raw[mask].copy()
    
    # 3. Run Integration Pipeline (GenoIntig)
    # Note: Ensure you use the specific pipeline function for the script you are editing
    # e.g., run_genointig_pipeline, run_scanorama_pipeline, etc.
    adata_neg = run_scanorama_pipeline(adata_neg) 
    
    # 4. Visualization (The Proof)
    # We highlight the 'target_cell_type' to show it stands apart
    sc.tl.umap(adata_neg)
    
    plt.figure(figsize=(8, 6))
    
    # Plot 1: Colored by Cell Type (Check separation)
    sc.pl.umap(adata_neg, color=CELLTYPE_KEY, show=False, title="Non-Overlapping Check (Cell Types)")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "Supp_NonOverlapping_CellTypes.png"), dpi=300)
    plt.close()
    
    # Plot 2: Colored by Batch (Check mixing of shared types vs isolation of unique type)
    sc.pl.umap(adata_neg, color=BATCH_KEY, show=False, title="Non-Overlapping Check (Batch)")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "Supp_NonOverlapping_Batch.png"), dpi=300)
    plt.close()
    
    logging.info("Non-Overlapping Experiment plots saved.")

def downsample_batch(adata, batch_key, target_batch, fraction):
    """
    Randomly subsamples a specific batch to the specified fraction.
    """
    adata_copy = adata.copy()
    mask = adata_copy.obs[batch_key] == target_batch
    idx = np.where(mask)[0]
    
    n_keep = int(len(idx) * fraction)
    if n_keep < 10: n_keep = len(idx) # Prevent empty batches
    
    keep_idx = np.random.choice(idx, n_keep, replace=False)
    remove_idx = np.setdiff1d(idx, keep_idx)
    
    # Keep everything that is NOT in remove_idx
    all_indices = np.arange(adata_copy.n_obs)
    final_indices = np.setdiff1d(all_indices, remove_idx)
    
    return adata_copy[final_indices].copy()

def load_data():
    """Loads and preprocesses the PBMC data."""
    try:
        adata = sc.read(DATA_FILE)
    except FileNotFoundError:
        logging.error(f"Data file not found at {DATA_FILE}")
        return None
    
    adata.obs_names_make_unique()

    if "highly_variable" in adata.var:
        adata = adata[:, adata.var["highly_variable"]].copy()
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var["highly_variable"]].copy()

    return adata 

def run_scanorama_pipeline(adata):
    """
    Runs the Scanorama integration pipeline on the provided AnnData.
    Returns the integrated AnnData object.
    """
    # 1. Split by Batch
    if BATCH_KEY not in adata.obs:
        logging.error(f"Batch key {BATCH_KEY} not found.")
        return adata
        
    adatas_list = [adata[adata.obs[BATCH_KEY] == batch].copy() for batch in adata.obs[BATCH_KEY].unique()]

    # 2. Run Scanorama
    # Note: hvg=2000 is standard
    try:
        corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True, hvg=2000)
    except Exception as e:
        logging.error(f"Scanorama failed: {e}")
        return adata

    # 3. Concatenate
    adata_int = ad.concat(corrected, join='outer', label="post_scanorama_batch_id")
    
    # 4. Restore Metadata (Labels)
    # We assume simple concatenation preserves order of blocks. 
    # A safer approach for robust scripts is mapping by index, but concat usually works here.
    if adata.n_obs == adata_int.n_obs:
        adata.obsm['X_scanorama'] = adata_int.obsm['X_scanorama']
    else:
        # If cells were dropped (rare in scanorama unless filtered), use the new object
        # and try to map labels
        adata = adata_int
        # This part is tricky if indices change. For this script, we assume stability.
    
    # 5. Clustering (Leiden)
    sc.pp.neighbors(adata, use_rep='X_scanorama', n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.5)
    
    return adata

# --- Main Analysis Function ---
def main():
    sc.settings.verbosity = 2
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    logging.info("Loading and preparing data...")
    adata_raw = load_data()
    if adata_raw is None: return
    
    if CELLTYPE_KEY not in adata_raw.obs.columns:
        logging.error(f"Required column '{CELLTYPE_KEY}' not in adata.obs.")
        return

    # --- 1. Full Integration Run ---
    logging.info("Running Scanorama on full dataset...")
    adata = run_scanorama_pipeline(adata_raw.copy())

    # Visualization
    logging.info("Generating plots...")
    sc.tl.umap(adata)
    sc.tl.tsne(adata, use_rep='X_scanorama')

    unique_labels = sorted(list(np.unique(adata.obs[CELLTYPE_KEY])))
    colors = cm.get_cmap('tab20', len(unique_labels))
    colors_list = [colors(i) for i in range(len(unique_labels))]
    label_to_color_map, _ = create_custom_colormap(unique_labels, colors_list)

    plot_filename_prefix = "Scanorama_Algorithm_iNMF_Dataset"
    plot_embedding(adata.obsm['X_umap'], adata.obs[CELLTYPE_KEY].astype(str), "UMAP", f"UMAP_Plot_{plot_filename_prefix}", custom_colors=label_to_color_map)
    plot_embedding(adata.obsm['X_tsne'], adata.obs[CELLTYPE_KEY].astype(str), "t-SNE", f"TSNE_Plot_{plot_filename_prefix}", custom_colors=label_to_color_map)

    # Evaluation Metrics
    logging.info("Calculating evaluation metrics...")
    ari, silhouette, ilisi, kbet = compute_all_metrics(adata)
    rand = rand_score(adata.obs[CELLTYPE_KEY], adata.obs['leiden'])
    
    logging.info(f"ARI: {ari:.3f}, Silhouette: {silhouette:.3f}, iLISI: {ilisi:.3f}, kBET: {kbet:.3f}")

    plot_metrics_bar(ari, rand, silhouette, plot_filename_prefix)
    plot_batch_metrics(ilisi, kbet, plot_filename_prefix)
    
    # Save CSV
    df = pd.DataFrame([{"ARI": ari, "Rand": rand, "Silhouette": silhouette, "iLISI": ilisi, "kBET": kbet}])
    df.to_csv(os.path.join(OUTPUT_FOLDER, f"CSV_{plot_filename_prefix}.csv"), index=False)

    # --- 2. Robustness (Downsampling) Analysis ---
    if RUN_ROBUSTNESS_ANALYSIS:
        logging.info("Starting Downsampling Analysis (Robustness)...")
        results = []
        
        # Identify the largest batch to downsample
        batch_counts = adata_raw.obs[BATCH_KEY].value_counts()
        batch_to_downsample = batch_counts.idxmax()
        logging.info(f"Targeting batch '{batch_to_downsample}' for downsampling.")

        for frac in [1.0, 0.5, 0.25, 0.1]:
            logging.info(f"Processing fraction: {frac}")
            
            # Downsample
            ad_ds = downsample_batch(adata_raw, BATCH_KEY, batch_to_downsample, frac)
            
            # Run Pipeline
            ad_ds = run_scanorama_pipeline(ad_ds)
            
            # Compute Metrics
            # Note: We skip kBET in loop for speed if desired, but here we include it
            ari_ds, sil_ds, ilisi_ds, kbet_ds = compute_all_metrics(ad_ds)
            
            results.append({
                "fraction": frac, 
                "ARI": ari_ds, 
                "Silhouette": sil_ds, 
                "iLISI": ilisi_ds, 
                "kBET": kbet_ds
            })
            logging.info(f"Frac {frac} -> ARI: {ari_ds:.3f}, iLISI: {ilisi_ds:.3f}")

        # Save and Plot Robustness
        df_imbalance = pd.DataFrame(results)
        df_imbalance.to_csv(os.path.join(OUTPUT_FOLDER, "Scanorama_imbalance_results.csv"), index=False)
        
        plt.figure(figsize=(6,5))
        for m in ['ARI', 'Silhouette', 'iLISI', 'kBET']:
            plt.plot(df_imbalance['fraction'], df_imbalance[m], label=m, marker='o')
        plt.xlabel("Fraction of retained cells")
        plt.ylabel("Score")
        plt.legend()
        plt.title("Scanorama Robustness to Imbalance")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(OUTPUT_FOLDER, "Scanorama_Imbalance_Robustness.png"), dpi=300)
        
    logging.info("Scanorama Analysis Complete.")
    # --- Non-Overlapping Experiment (Supplementary) ---
    # Pass the RAW original data (before any processing/integration)
    run_non_overlapping_experiment(adata_raw.copy())

if __name__ == "__main__":
    main()