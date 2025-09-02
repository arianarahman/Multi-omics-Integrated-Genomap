import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import scanorama
from sklearn.metrics import adjusted_rand_score, silhouette_score, rand_score

# --- Configuration ---
DATA_FILE = './Dataset/pbmcs_ctrl_converted.h5ad'
OUTPUT_FOLDER = './Figures_CSVs'
BATCH_KEY = 'batch'
CELLTYPE_KEY = 'celltype'

# --- Helper Functions (Standardized) ---
def plot_embedding(X, y, title, filename):
    """Generates and saves a 2D scatter plot of an embedding."""
    X, y = np.asarray(X), np.asarray(y)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet', s=18)
    plt.xlabel(f'{title} Dimension 1', fontsize=8)
    plt.ylabel(f'{title} Dimension 2', fontsize=8)
    plt.title(filename, fontsize=10)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{filename}.png'), dpi=300)
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

def save_metrics_csv(ari, rand, silhouette, filename):
    """Saves evaluation metrics to a CSV file."""
    df = pd.DataFrame([{"ARI": ari, "Rand": rand, "Silhouette": silhouette}])
    df.to_csv(os.path.join(OUTPUT_FOLDER, f'{filename}.csv'), index=False)

def load_and_prepare_data_for_scanorama():
    """
    Loads and preprocesses data using a unified workflow, then splits it by batch for Scanorama.
    """
    try:
        adata = sc.read(DATA_FILE)
    except FileNotFoundError:
        logging.error(f"Data file not found at {DATA_FILE}")
        return None, None

    # Safety check for the batch key
    if BATCH_KEY not in adata.obs.columns:
        logging.warning(f"'{BATCH_KEY}' not found in adata.obs. Creating a dummy batch ('batch1').")
        adata.obs[BATCH_KEY] = 'batch1'
        adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype('category')

    # --- Unified Preprocessing Workflow ---
    sc.pp.filter_cells(adata, min_genes=1)
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if np.isnan(adata.X.data).any():
        adata.X.data[np.isnan(adata.X.data)] = 0

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key=BATCH_KEY)
    adata = adata[:, adata.var.highly_variable]

    sc.pp.scale(adata, max_value=10)

    # --- Split data by batch for Scanorama ---
    adatas_list = [adata[adata.obs[BATCH_KEY] == b].copy() for b in adata.obs[BATCH_KEY].cat.categories]

    # Keep track of the original, full cell type labels
    true_labels_full = adata.obs[CELLTYPE_KEY].copy()

    return adatas_list, true_labels_full

# --- Main Analysis Function ---
def main():
    """Main function to run the Scanorama analysis workflow."""
    sc.settings.verbosity = 2
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    logging.info("Loading and preparing data for Scanorama...")
    adatas_list, true_labels_full = load_and_prepare_data_for_scanorama()
    if adatas_list is None:
        return

    # 2. Scanorama Integration
    logging.info("Running Scanorama integration...")
    adatas_corrected = scanorama.correct_scanpy(adatas_list, return_dimred=True, hvg=2000)

    # Concatenate the corrected results into a single object
    adata = sc.concat(adatas_corrected, join='outer', label="batch_id")
    adata.obs[CELLTYPE_KEY] = true_labels_full

    # 3. Clustering and Visualization
    logging.info("Building neighborhood graph, running Leiden, and generating embeddings...")
    sc.pp.neighbors(adata, use_rep='X_scanorama', n_neighbors=15)
    sc.tl.leiden(adata, resolution=0.8)

    sc.tl.umap(adata)
    sc.tl.tsne(adata, use_rep='X_scanorama')

    true_labels_cat = adata.obs[CELLTYPE_KEY].astype('category').cat.codes
    plot_embedding(adata.obsm['X_umap'], true_labels_cat, "UMAP", "UMAP_Plot_Scanorama_iNMF_Dataset")
    plot_embedding(adata.obsm['X_tsne'], true_labels_cat, "t-SNE", "TSNE_Plot_Scanorama_iNMF_Dataset")

    # 4. Evaluation
    logging.info("Calculating evaluation metrics...")
    true_labels = adata.obs[CELLTYPE_KEY].values
    predicted_labels = adata.obs['leiden'].values

    ari = adjusted_rand_score(true_labels, predicted_labels)
    rand = rand_score(true_labels, predicted_labels)
    silhouette = silhouette_score(adata.obsm['X_umap'], predicted_labels)

    logging.info(f"ARI: {ari:.3f}, Rand Index: {rand:.3f}, Silhouette Score: {silhouette:.3f}")

    plot_metrics_bar(ari, rand, silhouette, "Scanorama_Algorithm_With_iNMF_data")
    save_metrics_csv(ari, rand, silhouette, "CSV_Scanorama_Algorithm_With_iNMF_data")

    logging.info("Scanorama analysis complete.")

if __name__ == "__main__":
    main()