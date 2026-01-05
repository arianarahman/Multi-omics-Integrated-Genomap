"""
=============================================================================
Script Name:    GenoIntig_Pipeline_iNMF_Dataset.py
Author:         Ariana Rahman
Affiliation:    Arizona State University / Stanford University
Date:           January 2026
Description:    
    This script performs single-cell integration using GenoIntig (Scanorama + GenoMap)
    and evaluates its performance on PBMC datasets.

    Key Experiments:
    1. Full Integration & Visualization: UMAP/t-SNE, ARI, Silhouette.
    2. Batch Mixing Assessment: iLISI and kBET.
    3. Robustness Analysis: Downsampling the largest batch (100%, 50%, 25%, 10%).

    Output:
    - Generates 'UMAP/TSNE' plots.
    - Generates 'CSV' files with metrics.
    - Generates 'GenoIntig_Imbalance_Robustness.png'.
=============================================================================
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import scanpy as sc
import scanorama
from sklearn.metrics import adjusted_rand_score, silhouette_score, rand_score
from sklearn.neighbors import NearestNeighbors
from scib.metrics import kBET
import anndata as ad
import scipy.sparse
import genomap.genoDR as gp

# --- Configuration ---
DATA_FILE = './Dataset/pbmcs_ctrl_annotated.h5ad'
OUTPUT_FOLDER = './Figures_CSVs'
BATCH_KEY = 'batch'
CELLTYPE_KEY = 'celltype'

# --- Control Flags ---
CALCULATE_BATCH_METRICS = True
RUN_ROBUSTNESS_ANALYSIS = True

# --- Helper Functions ---

def create_custom_colormap(labels, colors):
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
            plt.scatter(X[indices, 0], X[indices, 1], color=custom_colors.get(label), label=label, s=18)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet', s=18)
    plt.title(filename, fontsize=15)
    plt.xlabel(f'{title} Dimension 1')
    plt.ylabel(f'{title} Dimension 2')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{filename}.png'), dpi=600)
    plt.close()

def plot_metrics_bar(ari, rand, silhouette, filename_prefix):
    metrics = ["ARI", "Rand Index", "Silhouette"]
    values = [ari, rand, silhouette]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['coral', 'lightgreen', 'blue'], width=0.3)
    plt.title(f"Metrics for {filename_prefix}")
    plt.ylim(0, 1)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f'{value:.2f}', ha='center')
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
    plt.title("Batch Mixing Performance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"BatchMetrics_{filename}.png"), dpi=300)
    plt.close()

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
    adata_neg = run_genointig_pipeline(adata_neg) 
    
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

def compute_lisi_python(X, labels, k=90):
    n_cells = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    unique_labels, encoded_labels = np.unique(labels, return_inverse=True)
    n_classes = len(unique_labels)
    lisi_scores = []
    for i in range(n_cells):
        counts = np.bincount(encoded_labels[indices[i]], minlength=n_classes)
        probs = counts / k
        simpson = np.sum(probs ** 2)
        lisi = 1 / simpson if simpson > 0 else 1
        lisi_scores.append(lisi)
    return np.mean(lisi_scores)

def compute_all_metrics(adata):
    ari = adjusted_rand_score(adata.obs[CELLTYPE_KEY], adata.obs['leiden'])
    try:
        silhouette = silhouette_score(adata.obsm['X_genomap'], adata.obs['leiden'])
    except:
        silhouette = 0.0
    
    ilisi, kbet = 0.0, 0.0
    if CALCULATE_BATCH_METRICS:
        try:
            ilisi = compute_lisi_python(adata.obsm['X_genomap'], adata.obs[BATCH_KEY], k=90)
        except Exception as e:
            logging.warning(f"iLISI failed: {e}")
        try:
            kbet_result = kBET(adata, batch_key=BATCH_KEY, label_key=CELLTYPE_KEY, type_="embed", embed="X_genomap", verbose=False)
            kbet = np.mean(kbet_result['accept'])
        except Exception as e:
            logging.warning(f"kBET failed: {e}")
            kbet = 0.0
    return ari, silhouette, ilisi, kbet

def downsample_batch(adata, batch_key, target_batch, fraction):
    adata_copy = adata.copy()
    mask = adata_copy.obs[batch_key] == target_batch
    idx = np.where(mask)[0]
    n_keep = max(10, int(len(idx) * fraction))
    keep_idx = np.random.choice(idx, n_keep, replace=False)
    remove_idx = np.setdiff1d(idx, keep_idx)
    all_indices = np.arange(adata_copy.n_obs)
    final_indices = np.setdiff1d(all_indices, remove_idx)
    return adata_copy[final_indices].copy()

def load_data():
    if not os.path.exists(DATA_FILE): return None
    adata = sc.read(DATA_FILE)
    adata.obs_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=1)
    if "highly_variable" not in adata.var:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata = adata[:, adata.var["highly_variable"]].copy()
    return adata 

def run_genointig_pipeline(adata):
    """Runs Scanorama + GenoMap integration pipeline."""
    # 1. Scanorama (Stage 1)
    if BATCH_KEY not in adata.obs:
        logging.error(f"Batch key {BATCH_KEY} not found.")
        return adata
        
    adatas_list = [adata[adata.obs[BATCH_KEY] == batch].copy() for batch in adata.obs[BATCH_KEY].unique()]
    
    try:
        corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True, hvg=2000)
    except Exception as e:
        logging.error(f"Scanorama failed: {e}")
        return adata
        
    adata_int = ad.concat(corrected, join='outer', label="post_scanorama_batch_id")
    
    # Check if we need to map indices back or if concat preserved them
    if adata.n_obs == adata_int.n_obs:
        adata.obsm['X_scanorama'] = adata_int.obsm['X_scanorama']
    else:
        adata = adata_int # Fallback if cells dropped

    # 2. GenoMap (Stage 2)
    # Determine n_clusters dynamically or default to 8
    n_clusters = len(adata.obs[CELLTYPE_KEY].unique()) if CELLTYPE_KEY in adata.obs else 8

    # Clean potential NaNs before passing to GenoDR
    if np.isnan(adata.obsm['X_scanorama']).any():
        adata.obsm['X_scanorama'] = np.nan_to_num(adata.obsm['X_scanorama'])

    adata.obsm['X_genomap'] = gp.genoDR(
        adata.obsm['X_scanorama'], 
        n_dim=32, 
        n_clusters=n_clusters, 
        colNum=33, 
        rowNum=33
    )

    # 3. Clustering & Visualization
    sc.pp.neighbors(adata, use_rep='X_genomap', n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.5)
    sc.tl.umap(adata)
    sc.tl.tsne(adata, use_rep='X_genomap')
    
    return adata

def main():
    sc.settings.verbosity = 0
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    logging.info("Loading data...")
    adata_raw = load_data()
    if adata_raw is None: return

    # --- 1. Full Integration ---
    logging.info("Running GenoIntig on full dataset...")
    adata = run_genointig_pipeline(adata_raw.copy())

    # Plots
    unique_labels = sorted(list(np.unique(adata.obs[CELLTYPE_KEY])))
    colors = cm.get_cmap('tab20', len(unique_labels))
    label_to_color_map, _ = create_custom_colormap(unique_labels, [colors(i) for i in range(len(unique_labels))])
    
    prefix = "GenoIntig_Pipeline_iNMF_Dataset"
    plot_embedding(adata.obsm['X_umap'], adata.obs[CELLTYPE_KEY], "UMAP", f"UMAP_{prefix}", custom_colors=label_to_color_map)
    plot_embedding(adata.obsm['X_tsne'], adata.obs[CELLTYPE_KEY], "t-SNE", f"TSNE_{prefix}", custom_colors=label_to_color_map)

    # Metrics
    ari, silhouette, ilisi, kbet = compute_all_metrics(adata)
    rand = rand_score(adata.obs[CELLTYPE_KEY], adata.obs['leiden'])
    logging.info(f"Full Run -> ARI: {ari:.3f}, iLISI: {ilisi:.3f}")
    
    plot_metrics_bar(ari, rand, silhouette, prefix)
    plot_batch_metrics(ilisi, kbet, prefix)
    pd.DataFrame([{"ARI": ari, "Rand": rand, "Silhouette": silhouette, "iLISI": ilisi, "kBET": kbet}]).to_csv(os.path.join(OUTPUT_FOLDER, f"CSV_{prefix}.csv"), index=False)

    # --- 2. Robustness Analysis ---
    if RUN_ROBUSTNESS_ANALYSIS:
        logging.info("Starting Robustness Analysis...")
        results = []
        batch_to_downsample = adata_raw.obs[BATCH_KEY].value_counts().idxmax()

        for frac in [1.0, 0.5, 0.25, 0.1]:
            logging.info(f"Processing fraction: {frac}")
            ad_ds = downsample_batch(adata_raw, BATCH_KEY, batch_to_downsample, frac)
            ad_ds = run_genointig_pipeline(ad_ds)
            
            ari_ds, sil_ds, ilisi_ds, kbet_ds = compute_all_metrics(ad_ds)
            results.append({"fraction": frac, "ARI": ari_ds, "Silhouette": sil_ds, "iLISI": ilisi_ds, "kBET": kbet_ds})
            logging.info(f"Frac {frac} -> ARI: {ari_ds:.3f}")

        df_imb = pd.DataFrame(results)
        df_imb.to_csv(os.path.join(OUTPUT_FOLDER, "GenoIntig_imbalance_results.csv"), index=False)
        
        plt.figure(figsize=(6,5))
        for m in ['ARI', 'Silhouette', 'iLISI', 'kBET']:
            plt.plot(df_imb['fraction'], df_imb[m], label=m, marker='o')
        plt.legend()
        plt.title("GenoIntig Robustness to Imbalance")
        plt.savefig(os.path.join(OUTPUT_FOLDER, "GenoIntig_Imbalance_Robustness.png"), dpi=300)

        # --- Non-Overlapping Experiment (Supplementary) ---
        # Pass the RAW original data (before any processing/integration)
        run_non_overlapping_experiment(adata_raw.copy())

if __name__ == "__main__":
    main()